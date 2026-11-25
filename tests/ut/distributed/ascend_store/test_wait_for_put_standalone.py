"""Stdlib-only tests for the released-request save fence timing (worker_summary stage wait_for_put).

Set VLLM_COMPACT_ROOT to the vLLM checkout that carries vllm/v1/metrics/pd_diagnostics.py.
The fence method is compiled from kv_transfer.py's source, so neither torch nor NPU packages load.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import logging
import os
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VLLM_ROOT = Path(os.environ.get("VLLM_COMPACT_ROOT", ROOT.parent / "vllm"))
KV_TRANSFER = ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = load_source("wait_for_put_diag", VLLM_ROOT / "vllm/v1/metrics/pd_diagnostics.py")


def fence_function():
    tree = ast.parse(KV_TRANSFER.read_text())
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "KVCacheStoreSendingThread":
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name == "wait_for_requests_saved":
                    module = ast.Module(body=[node], type_ignores=[])
                    namespace = {
                        "diag": diag,
                        "time": time,
                        "logger": logging.getLogger("wait_for_put_test"),
                        "SAVE_REQUEST_POLL_INTERVAL_S": 0.001,
                    }
                    exec(compile(module, str(KV_TRANSFER), "exec"), namespace)
                    return namespace["wait_for_requests_saved"]
    raise AssertionError("wait_for_requests_saved not found")


WAIT = fence_function()


class FakeSendThread:
    def __init__(self, stored):
        self.done_task_lock = threading.Lock()
        self.stored_requests = dict(stored)
        self.tp_rank = 0
        self.fail = None

    def raise_if_failed(self):
        if self.fail is not None:
            raise self.fail


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("[PD_DIAG] "):
            self.records.append(json.loads(message[len("[PD_DIAG] ") :]))


class WaitForPutTest(unittest.TestCase):
    def setUp(self):
        self.capture = Capture()
        logger = logging.getLogger(diag.__name__)
        logger.addHandler(self.capture)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    def tearDown(self):
        logging.getLogger(diag.__name__).removeHandler(self.capture)

    def run_step(self, thread, released):
        with diag.batch("worker_summary", phase="execute", execution_id="execution-x-1-7", request_ids=["r2"], tp_rank=0):
            WAIT(thread, released)
        (record,) = [r for r in self.capture.records if r["event"] == "worker_summary"]
        return record

    def test_nothing_to_wait_records_a_stage(self):
        record = self.run_step(FakeSendThread({}), set())
        stage = record["stages"]["wait_for_put"]
        self.assertEqual(stage["calls"], 1)
        self.assertEqual(stage["measured_calls"], 1)
        self.assertEqual(stage["released_requests"], 0)
        self.assertEqual(stage["status"], "ok")
        self.assertEqual(stage["timing_scope"], "released_request_save_fence")
        self.assertGreaterEqual(stage["duration_ns"], 0)
        self.assertEqual(record["execution_id"], "execution-x-1-7")
        self.assertEqual(record["request_ids"], ["r2"])
        self.assertEqual(record["tp_rank"], 0)

    def test_waits_until_released_saves_finish(self):
        thread = FakeSendThread({"r1": {1}})

        def finish():
            time.sleep(0.03)
            with thread.done_task_lock:
                thread.stored_requests.pop("r1")

        worker = threading.Thread(target=finish)
        worker.start()
        record = self.run_step(thread, {"r1", "r0"})
        worker.join()
        stage = record["stages"]["wait_for_put"]
        self.assertEqual(stage["released_requests"], 2)
        self.assertGreaterEqual(stage["duration_ns"], 25_000_000)
        self.assertEqual(stage["duration_ns"], stage["end_ns"] - stage["start_ns"])

    def test_failure_is_recorded_and_raised(self):
        thread = FakeSendThread({"r1": {1}})
        thread.fail = RuntimeError("save thread failed")
        with self.assertRaises(RuntimeError):
            self.run_step(thread, {"r1"})
        (record,) = [r for r in self.capture.records if r["event"] == "worker_summary"]
        stage = record["stages"]["wait_for_put"]
        self.assertEqual(stage["status"], "error")
        self.assertEqual(stage["exceptions"], 1)

    def test_without_worker_summary_nothing_is_written(self):
        WAIT(FakeSendThread({}), {"r9"})
        self.assertEqual(self.capture.records, [])


if __name__ == "__main__":
    unittest.main()
