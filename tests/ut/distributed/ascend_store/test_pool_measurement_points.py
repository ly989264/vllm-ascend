"""Each Pool measurement point records exactly the work it names (vllm-ascend d4ad49796).

The code under test is the real ChunkedTokenDatabase timing wrappers, KVCacheStoreSendingThread,
KVPoolWorker, AscendStoreConnector and MooncakeBackend; only the Mooncake store object, NPU events
and the innermost key/addr helpers are fakes. Every fake step sleeps for a distinct, known time, so a
stage that included another stage's work would exceed its own bound.
Needs the real vllm pd_diagnostics module (VLLM_COMPACT_ROOT plugin) to see the records.
"""

import json
import threading
import time
import unittest
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import diagnostics as pool_diag
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_backend as mb
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

MS = 1_000_000
diag = pool_diag.diag
TERMINAL_LOADED = {"succeeded", "backend_failure", "unknown_result", "exception"}


def sleep_ms(value):
    if value:
        time.sleep(value / 1000)


class TimedTokenDatabase(ChunkedTokenDatabase):
    """Real (wrapped) prepare_value and process_token_key_strings_with_block_ids; fake inner helpers."""

    def __init__(self, key_ms=0, prepare_ms=0, chunks=2, raise_in_prepare=False):  # noqa: super-init-not-called
        self.key_ms = key_ms
        self.prepare_ms = prepare_ms
        self.chunks = chunks
        self.raise_in_prepare = raise_in_prepare
        self.group_block_len = {0: [8]}
        self.group_kv_caches_base_addr = {0: [1000]}
        self.group_block_stride = {}
        self.group_num_layers = {"kv": {0: 1}}
        self.partitions = None
        self.hash_block_size = 16

    def store_mask(self, token_len, num_prompt_tokens):
        return None

    def load_mask(self, block_hashes, token_len):
        return None

    def mask_allows_chunk(self, masks, group_id, start):
        return True

    def _get_key_prefix(self, kv_cache_group_id=0):
        return "model@"

    def _iter_token_chunks(self, token_len, block_hashes, mask_num=0, kv_cache_group_id=0, *args, **kwargs):
        block_ids = kwargs.get("block_ids") or [0] * self.chunks
        for index in range(min(self.chunks, token_len // 16)):
            sleep_ms(self.key_ms)
            yield index * 16, (index + 1) * 16, f"h{index}", block_ids[index]

    def get_block_size(self, kv_cache_group_id=0):
        sleep_ms(self.prepare_ms)
        if self.raise_in_prepare:
            raise RuntimeError("prepare sentinel")
        return 16


class FakeNative:
    """The Mooncake store object inside MooncakeBackend; records when and on which thread it ran."""

    def __init__(self, exists_ms=0, put_ms=0, get_ms=0, get_result=None, get_raises=False):
        self.exists_ms, self.put_ms, self.get_ms = exists_ms, put_ms, get_ms
        self.get_result = get_result
        self.get_raises = get_raises
        self.calls = []

    def _record(self, name, start_ns, start_wall_ns, end_ns, end_wall_ns, **extra):
        self.calls.append(
            dict(
                name=name,
                start_ns=start_ns,
                end_ns=end_ns,
                start_wall_ns=start_wall_ns,
                end_wall_ns=end_wall_ns,
                tid=threading.get_native_id(),
                **extra,
            )
        )

    def _timed(self, name, ms, **extra):
        start_ns, start_wall_ns = time.monotonic_ns(), time.time_ns()
        sleep_ms(ms)
        self._record(name, start_ns, start_wall_ns, time.monotonic_ns(), time.time_ns(), **extra)

    def batch_is_exist(self, keys):
        self._timed("exists", self.exists_ms)
        return [0] * len(keys)

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config):
        self._timed("put", self.put_ms, addrs=addrs)
        return [sum(size) for size in sizes]

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        self._timed("get", self.get_ms, addrs=addrs, keys=list(keys))
        if self.get_raises:
            raise RuntimeError("native get sentinel")
        if self.get_result is not None:
            return self.get_result(keys)
        return [sum(size) for size in sizes]


def make_backend(native, initialized=True):
    backend = object.__new__(mb.MooncakeBackend)
    backend.store = native
    backend._lazy_init = not initialized
    backend._store_initialized = initialized
    backend.ensure_initialized = lambda: None
    backend._build_replicate_config = lambda: None
    backend.set_device = lambda: None
    return backend


def records(captured, event=None):
    out = [json.loads(r.getMessage().split("[PD_DIAG] ", 1)[1]) for r in captured.records]
    return [r for r in out if event is None or r["event"] == event]


def load_request(req_id, load_spec="hit", tokens=32):
    spec = None
    if load_spec == "hit":
        spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=tokens, can_load=True)
    elif load_spec == "no_load":
        spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=tokens, can_load=False)
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=tokens,
        block_ids=[1, 2],
        block_hashes=["h0", "h1"],
        load_spec=spec,
        is_last_chunk=True,
    )


def sync_load_worker(backend, database=None, tp_mismatch=False):
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.tp_rank = 0
    worker.use_layerwise = False
    worker.load_async = False
    worker.tp_mismatch = tp_mismatch
    worker.cache_transfer_granularity = 16
    worker.group_uses_align_state = []
    worker.token_database = database or TimedTokenDatabase()
    worker.grouped_block_size = [16]
    worker.m_store = backend
    worker._invalid_block_ids = set()
    worker._invalid_block_ids_lock = threading.Lock()
    worker._record_kv_connector_operation = lambda *args: None
    worker.kv_recv_thread = None
    return worker


def metadata(execution_id, *requests):
    meta = AscendConnectorMetadata(set(), set(), pd_execution_id=execution_id)
    for request in requests:
        meta.add_request(request)
    return meta


class TestSyncLoadRecords(unittest.TestCase):
    """A prefill node with load_async=false: start_load_kv writes one pool_task[load] per request."""

    def test_each_request_is_one_load_task_that_contains_its_own_get(self):
        native = FakeNative(get_ms=10)
        worker = sync_load_worker(make_backend(native))
        requests = [load_request(f"r{index}") for index in range(3)]
        with self.assertLogs(diag._LOG, level="INFO") as captured:
            worker.start_load_kv(metadata("exec-sync", *requests))
        tasks = records(captured, "pool_task")
        self.assertEqual([t["operation"] for t in tasks], ["load"] * 3)
        self.assertEqual([t["request_id"] for t in tasks], ["r0", "r1", "r2"])
        gets = [call for call in native.calls if call["name"] == "get"]
        self.assertEqual(len(gets), 3)
        for task, get in zip(tasks, gets):
            self.assertEqual(task["status"], "succeeded")
            self.assertEqual(task["execution_id"], "exec-sync")
            self.assertEqual(task["member_count"], 1)
            self.assertEqual(task["tp_rank"], 0)
            self.assertIsNone(task["queue_enter_ns"])
            self.assertNotIn("queue_wait_ns", task["counters"])
            self.assertNotIn("queue_entries", task["counters"])
            # The native get (it copies straight into the KV cache addresses below) ran inside this task.
            self.assertLessEqual(task["start_ns"], get["start_ns"])
            self.assertLessEqual(get["end_ns"], task["end_ns"])
            self.assertGreaterEqual(task["duration_ns"], get["end_ns"] - get["start_ns"])
            self.assertEqual(get["keys"], ["model@h0", "model@h1"])
            self.assertEqual(get["addrs"], [[1000 + 1 * 8], [1000 + 2 * 8]])
            self.assertEqual(task["associated_stages"]["backend_get"]["calls"], 1)
            self.assertEqual(task["native_calls"][0]["operation"], "mooncake_get")
        # Tasks are sequential and each holds only its own get.
        for earlier, later in zip(tasks, tasks[1:]):
            self.assertLessEqual(earlier["end_ns"], later["start_ns"])

    def test_tp_mismatch_load_is_inside_the_task(self):
        native = FakeNative(get_ms=10)
        worker = sync_load_worker(make_backend(native), tp_mismatch=True)

        def keys_and_addrs(block_hashes, block_ids, token_len, mask_num, **kwargs):
            return ["k0", "k1"], [[1100], [1200]], [[8], [8]], [1, 2]

        worker._build_tp_mismatch_keys_and_addrs = keys_and_addrs
        with self.assertLogs(diag._LOG, level="INFO") as captured:
            worker.start_load_kv(metadata("exec-tpm", load_request("r-tpm")))
        (task,) = records(captured, "pool_task")
        get = native.calls[0]
        self.assertEqual((task["operation"], task["status"], task["request_id"]), ("load", "succeeded", "r-tpm"))
        self.assertLessEqual(task["start_ns"], get["start_ns"])
        self.assertLessEqual(get["end_ns"], task["end_ns"])

    def test_terminal_status_of_every_request_kind(self):
        def run(request, native=None, database=None, initialized=True):
            native = native or FakeNative()
            worker = sync_load_worker(make_backend(native, initialized), database)
            with self.assertLogs(diag._LOG, level="INFO") as captured:
                try:
                    worker.start_load_kv(metadata("exec-status", request))
                    raised = None
                except RuntimeError as error:
                    raised = error
            (task,) = records(captured, "pool_task")
            self.assertEqual((task["operation"], task["request_id"]), ("load", request.req_id))
            return task, raised

        cases = {
            "missing_load_spec": lambda: run(load_request("r", load_spec=None)),
            "skipped": lambda: run(load_request("r", load_spec="no_load")),
            "no_task": lambda: run(load_request("r", tokens=0)),
            "succeeded": lambda: run(load_request("r")),
            "backend_failure (a key failed)": lambda: run(
                load_request("r"), FakeNative(get_result=lambda keys: [8, -704][: len(keys)])
            ),
            "backend_failure (native get raised, backend returned None)": lambda: run(
                load_request("r"), FakeNative(get_raises=True)
            ),
            "unknown_result (store not initialized, no native call)": lambda: run(load_request("r"), initialized=False),
            "exception (raised outside the backend)": lambda: run(
                load_request("r"), database=TimedTokenDatabase(raise_in_prepare=True)
            ),
        }
        for label, case in cases.items():
            with self.subTest(case=label):
                task, raised = case()
                expected = label.split(" ", 1)[0]
                self.assertEqual(task["status"], expected)
                if expected == "exception":
                    self.assertIsNotNone(raised)
                    self.assertEqual(str(raised), "prepare sentinel")
                else:
                    self.assertIsNone(raised)
                if expected in ("missing_load_spec", "skipped", "no_task"):
                    self.assertNotIn("backend_get", task["associated_stages"])
                if "native get raised" in label:
                    self.assertEqual(task["counters"]["mooncake_get_exceptions"], 1)
                    self.assertEqual(task["counters"]["get_unknown_result_keys"], 2)
                if "not initialized" in label:
                    self.assertEqual(task["native_calls"], [])
                    self.assertEqual(task["counters"]["get_unknown_result_keys"], 2)

    def test_request_durations_add_up_to_the_step_submit_stage(self):
        """Synthetic step: vLLM's kv.load_submit span around start_load_kv, as in the KV connector mixin."""
        native = FakeNative(get_ms=10)
        worker = sync_load_worker(make_backend(native))
        requests = [load_request("hit-0"), load_request("miss", load_spec=None), load_request("hit-1")]
        requests += [load_request("skip", load_spec="no_load"), load_request("hit-2")]
        context = {"execution_id": "exec-step", "request_id": None, "request_ids": [r.req_id for r in requests]}
        with self.assertLogs(diag._LOG, level="INFO") as captured:
            with diag.scope(**context), diag.batch("worker_summary", phase="execute", **context):
                with diag.span("kv.load_submit", completion="submission_return_not_transfer_completion"):
                    worker.start_load_kv(metadata("exec-step", *requests))
        tasks = records(captured, "pool_task")
        (step,) = records(captured, "worker_summary")
        submit = step["stages"]["submit"]
        self.assertEqual(len(tasks), len(requests))
        total = sum(task["duration_ns"] for task in tasks)
        loaded = sum(task["duration_ns"] for task in tasks if task["status"] in TERMINAL_LOADED)
        self.assertEqual(sorted(t["status"] for t in tasks), ["missing_load_spec", "skipped"] + ["succeeded"] * 3)
        overhead = submit["duration_ns"] - total
        print(
            f"\nsync-load step: submit={submit['duration_ns'] / MS:.3f} ms, sum(all tasks)={total / MS:.3f} ms, "
            f"sum(selected statuses)={loaded / MS:.3f} ms, outside tasks={overhead / MS:.3f} ms, requests={len(tasks)}"
        )
        # Every task lies inside the step's submit span, so the tasks can never add up to more.
        self.assertGreaterEqual(overhead, 0)
        for task in tasks:
            self.assertLessEqual(submit["start_ns"], task["start_ns"])
            self.assertLessEqual(task["end_ns"], submit["end_ns"])
        # What the tasks leave out is per-request capture/record writing and the function's own entry.
        # It is shorter than one 10 ms get, so no get ran outside a task.
        self.assertLess(overhead, 10 * MS)
        self.assertGreaterEqual(loaded, 3 * 10 * MS)
        for task in tasks:
            if task["status"] not in TERMINAL_LOADED:
                self.assertEqual(task["native_calls"], [])
                self.assertLess(task["duration_ns"], 10 * MS)


class TestSaveRecords(unittest.TestCase):
    """Background save: pool_task[save] and its five sub-stages."""

    def test_each_save_stage_measures_only_its_own_call(self):
        # Distinct durations (ms): build_key 2 x 10, prepare 2 x 15, lookup 40, sync 50, put 60; queue 30.
        native = FakeNative(exists_ms=40, put_ms=60)
        thread = KVCacheStoreSendingThread(make_backend(native), TimedTokenDatabase(key_ms=10, prepare_ms=15), 16, 0, 1)
        event = SimpleNamespace(synchronize=lambda: sleep_ms(50))
        request = ReqMeta(
            req_id="r-save",
            token_len_chunk=32,
            block_ids=[1, 2],
            block_hashes=["h0", "h1"],
            can_save=True,
            current_event=event,
            pd_execution_id="exec-save",
        )
        with self.assertLogs(diag._LOG, level="INFO") as captured:
            thread.add_save_batch([request])
            sleep_ms(30)
            while not thread.request_queue.empty():
                thread._dispatch_diagnostic_request(thread.request_queue.get())
        (task,) = records(captured, "pool_task")
        self.assertEqual((task["operation"], task["status"], task["request_id"]), ("save", "succeeded", "r-save"))
        self.assertEqual(task["execution_id"], "exec-save")
        expected = {"build_key": 20, "prepare": 30, "lookup": 40, "sync": 50, "put": 60}
        stages = task["stages"]
        for name, floor_ms in expected.items():
            with self.subTest(stage=name):
                self.assertEqual(stages[name]["exceptions"], 0)
                self.assertGreaterEqual(stages[name]["duration_ns"], floor_ms * MS)
                self.assertLessEqual(stages[name]["duration_ns"], stages[name]["end_ns"] - stages[name]["start_ns"])
        # The save runs build_key -> lookup -> prepare -> sync -> put; each stage's window ends before
        # the next one starts, so no stage holds another stage's work.
        order = ["build_key", "lookup", "prepare", "sync", "put"]
        for earlier, later in zip(order, order[1:]):
            self.assertLessEqual(stages[earlier]["end_ns"], stages[later]["start_ns"], (earlier, later))
        exists = next(c for c in native.calls if c["name"] == "exists")
        put = next(c for c in native.calls if c["name"] == "put")
        self.assertLessEqual(stages["lookup"]["start_ns"], exists["start_ns"])
        self.assertLessEqual(exists["end_ns"], stages["lookup"]["end_ns"])
        self.assertLessEqual(stages["put"]["start_ns"], put["start_ns"])
        self.assertLessEqual(put["end_ns"], stages["put"]["end_ns"])
        self.assertLessEqual(task["start_ns"], stages["build_key"]["start_ns"])
        self.assertLessEqual(stages["put"]["end_ns"], task["end_ns"])
        self.assertEqual(task["stages"]["prepare"]["calls"], 2)
        self.assertEqual(task["stages"]["put"]["calls"], 1)
        self.assertGreaterEqual(task["duration_ns"], sum(expected.values()) * MS)
        self.assertGreaterEqual(task["counters"]["queue_wait_ns"], 30 * MS)
        self.assertEqual(task["counters"]["queue_entries"], 1)

    def test_mooncake_put_native_window_matches_the_native_call(self):
        native = FakeNative(put_ms=20)
        thread = KVCacheStoreSendingThread(make_backend(native), TimedTokenDatabase(), 16, 0, 1)
        request = ReqMeta(
            req_id="r-put", token_len_chunk=32, block_ids=[1, 2], block_hashes=["h0", "h1"], can_save=True
        )
        with self.assertLogs(diag._LOG, level="INFO") as captured:
            thread.add_save_batch([request])
            while not thread.request_queue.empty():
                thread._dispatch_diagnostic_request(thread.request_queue.get())
        (task,) = records(captured, "pool_task")
        (call,) = task["native_calls"]
        put = next(c for c in native.calls if c["name"] == "put")
        self.assertEqual(call["operation"], "mooncake_put")
        self.assertEqual(call["native_tid"], put["tid"])
        self.assertLessEqual(call["start_wall_ns"], put["start_wall_ns"])
        self.assertLessEqual(put["end_wall_ns"], call["end_wall_ns"])
        self.assertLessEqual(call["start_ns"], put["start_ns"])
        self.assertLessEqual(put["end_ns"], call["end_ns"])
        self.assertEqual((call["submitted_keys"], call["submitted_bytes"], call["status"]), (2, 16, "succeeded"))
        self.assertGreaterEqual(task["stages"]["put"]["duration_ns"], call["duration_ns"])


class TestWaitForPutChain(unittest.TestCase):
    """Runner -> AscendStoreConnector.handle_preemptions -> handle_released_saves -> wait_for_requests_saved."""

    def run_step(self, released, release_after_ms):
        thread = KVCacheStoreSendingThread(make_backend(FakeNative()), TimedTokenDatabase(), 16, 0, 1)
        thread.stored_requests["r-old"] = 1
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.kv_send_thread = thread
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        connector.connector_worker = worker
        meta = AscendConnectorMetadata(set(), set(), pd_execution_id="exec-fence")
        meta.released_req_ids = set(released)

        def finish_save():
            sleep_ms(release_after_ms)
            with thread.done_task_lock:
                del thread.stored_requests["r-old"]

        finisher = threading.Thread(target=finish_save)
        context = {"execution_id": "exec-fence", "request_id": None, "request_ids": ["r-new"], "tp_rank": 0}
        with self.assertLogs(diag._LOG, level="INFO") as captured:
            finisher.start()
            with diag.scope(**context), diag.batch("worker_summary", phase="execute", **context):
                connector.handle_preemptions(meta)
        finisher.join()
        (step,) = records(captured, "worker_summary")
        return step

    def test_wait_is_recorded_in_the_step_record(self):
        step = self.run_step({"r-old"}, 20)
        stage = step["stages"]["wait_for_put"]
        self.assertEqual((step["execution_id"], step["request_ids"]), ("exec-fence", ["r-new"]))
        self.assertEqual(stage["released_requests"], 1)
        self.assertEqual(stage["timing_scope"], "released_request_save_fence")
        self.assertEqual((stage["calls"], stage["status"], stage["exceptions"]), (1, "ok", 0))
        self.assertGreaterEqual(stage["duration_ns"], 20 * MS)

    def test_nothing_to_wait_still_records_the_stage(self):
        step = self.run_step(set(), 0)
        stage = step["stages"]["wait_for_put"]
        self.assertEqual(stage["released_requests"], 0)
        self.assertEqual((stage["calls"], stage["status"]), (1, "ok"))
        # Only the function's own entry and exit: far below one 1 ms poll interval under normal load.
        self.assertLess(stage["duration_ns"], 5 * MS)


if __name__ == "__main__":
    unittest.main()
