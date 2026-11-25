"""What the Pool load records cover with load_async on and off.

With load_async=false the worker fetches the KV inside start_load_kv, so the one pool_task it writes
(operation "load", not "load_submit") includes the backend get. With load_async=true start_load_kv only
queues the request (pool_task "load_submit"); the get runs later in the receive thread (pool_task "load").
Needs the real vllm pd_diagnostics module (VLLM_COMPACT_ROOT plugin) to see the records.
"""

import json
import time
import unittest
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import diagnostics as pool_diag
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata, LoadSpec, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

GET_DELAY_S = 0.005


class SlowStore:
    requires_exists_before_put = True

    def __init__(self):
        self.gets = 0

    def get(self, keys, addrs, sizes):
        self.gets += 1
        time.sleep(GET_DELAY_S)
        return [0] * len(keys)


class TokenDatabase:
    group_block_len = {0: [8]}
    group_num_layers = {"kv": {0: 1}}

    def load_mask(self, block_hashes, token_len):
        return None

    def mask_allows_chunk(self, masks, group_id, start):
        return True

    def process_token_key_strings_with_block_ids(self, token_len, block_hashes, block_ids, mask_num=0, **kwargs):
        for index, block_id in enumerate(block_ids):
            start, end = index * 16, (index + 1) * 16
            if end <= token_len and start >= mask_num:
                yield start, end, f"key-{index}", f"hash-{index}", block_id

    def prepare_value(self, start, end, block_ids, kv_cache_group_id=0, cache_role="kv", block_id=None):
        return [1000 + start], [8], block_id


def make_worker(load_async: bool, store: SlowStore) -> KVPoolWorker:
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.tp_rank = 0
    worker.use_layerwise = False
    worker.load_async = load_async
    worker.tp_mismatch = False
    worker.cache_transfer_granularity = 16
    worker.group_uses_align_state = []
    worker.token_database = TokenDatabase()
    worker.grouped_block_size = [16]
    worker.m_store = store
    worker._invalid_block_ids = set()
    worker._record_kv_connector_operation = lambda *args: None
    worker.kv_recv_thread = KVCacheStoreRecvingThread(store, TokenDatabase(), 16, 0, 1)
    return worker


def make_metadata(execution_id: str) -> AscendConnectorMetadata:
    meta = AscendConnectorMetadata(set(), set(), pd_execution_id=execution_id)
    meta.add_request(
        ReqMeta(
            req_id="r-load",
            token_len_chunk=32,
            block_ids=[1, 2],
            block_hashes=["h0", "h1"],
            load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True),
            is_last_chunk=True,
        )
    )
    return meta


def pool_tasks(captured) -> list[dict]:
    records = [json.loads(r.getMessage().split("[PD_DIAG] ", 1)[1]) for r in captured.records]
    return [record for record in records if record["event"] == "pool_task"]


class TestPoolLoadScope(unittest.TestCase):
    def test_sync_load_writes_one_load_task_that_includes_the_get(self):
        store = SlowStore()
        worker = make_worker(False, store)
        with self.assertLogs(pool_diag.diag._LOG, level="INFO") as captured:
            worker.start_load_kv(make_metadata("exec-sync"))
        tasks = pool_tasks(captured)
        self.assertEqual([task["operation"] for task in tasks], ["load"])
        task = tasks[0]
        self.assertEqual(store.gets, 1)
        self.assertEqual(task["scope"], "task")
        self.assertEqual(task["status"], "succeeded")
        self.assertEqual(task["request_id"], "r-load")
        self.assertEqual(task["execution_id"], "exec-sync")
        self.assertIsNone(task["queue_enter_ns"])
        self.assertNotIn("queue_wait_ns", task["counters"])
        self.assertNotIn("queue_entries", task["counters"])
        self.assertEqual(task["counters"]["get_submitted_keys"], 2)
        self.assertEqual(task["associated_stages"]["backend_get"]["calls"], 1)
        # The get ran inside this task, so its duration is part of the task's.
        self.assertGreaterEqual(task["associated_stages"]["backend_get"]["duration_ns"], GET_DELAY_S * 1e9)
        self.assertGreaterEqual(task["duration_ns"], task["associated_stages"]["backend_get"]["duration_ns"])
        self.assertTrue(worker.kv_recv_thread.request_queue.empty())

    def test_async_load_submit_only_queues_and_the_load_task_carries_the_get(self):
        store = SlowStore()
        worker = make_worker(True, store)
        with self.assertLogs(pool_diag.diag._LOG, level="INFO") as captured:
            worker.start_load_kv(make_metadata("exec-async"))
            self.assertEqual(store.gets, 0)
            thread = worker.kv_recv_thread
            thread._dispatch_diagnostic_request(thread.request_queue.get())
        submit, load = pool_tasks(captured)
        self.assertEqual((submit["operation"], load["operation"]), ("load_submit", "load"))
        self.assertEqual(submit["status"], "submission_returned")
        self.assertIsNone(submit["queue_enter_ns"])
        self.assertNotIn("backend_get", submit["associated_stages"])
        self.assertLess(submit["duration_ns"], GET_DELAY_S * 1e9)
        self.assertEqual(load["status"], "succeeded")
        self.assertEqual((load["request_id"], load["execution_id"]), ("r-load", "exec-async"))
        self.assertIsInstance(load["queue_enter_ns"], int)
        self.assertEqual(load["counters"]["queue_entries"], 1)
        self.assertGreaterEqual(load["counters"]["queue_wait_ns"], 0)
        self.assertEqual(load["associated_stages"]["backend_get"]["calls"], 1)
        self.assertGreaterEqual(load["duration_ns"], GET_DELAY_S * 1e9)

    def test_pool_load_link_only_for_async_loads(self):
        for load_async in (False, True):
            with self.subTest(load_async=load_async):
                scheduler = object.__new__(KVPoolScheduler)
                scheduler.load_async = load_async
                scheduler.use_layerwise = False
                scheduler.vllm_config = SimpleNamespace(
                    kv_transfer_config=SimpleNamespace(engine_id="0"),
                    parallel_config=SimpleNamespace(data_parallel_rank=0),
                )
                emit = lambda: scheduler._emit_load_links(  # noqa: E731
                    make_metadata("exec-link"), SimpleNamespace(pd_execution_id="exec-link")
                )
                if not load_async:
                    with self.assertNoLogs(pool_diag.diag._LOG, level="INFO"):
                        emit()
                    continue
                with self.assertLogs(pool_diag.diag._LOG, level="INFO") as captured:
                    emit()
                records = [json.loads(r.getMessage().split("[PD_DIAG] ", 1)[1]) for r in captured.records]
                self.assertEqual(len(records), 1)
                link = records[0]
                self.assertEqual((link["event"], link["scope"], link["transfer_role"]), ("pd_transfer_summary", "link", "pool_load"))
                self.assertEqual((link["request_id"], link["execution_id"]), ("r-load", "exec-link"))


if __name__ == "__main__":
    unittest.main()
