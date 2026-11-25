"""Stdlib-only tests for the pool-load link record written by KVPoolScheduler.build_connector_meta.

Set VLLM_COMPACT_ROOT to the vLLM checkout that carries vllm/v1/metrics/pd_diagnostics.py.
The method is compiled from pool_scheduler.py's source, so neither torch nor vllm packages load.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
VLLM_ROOT = Path(os.environ.get("VLLM_COMPACT_ROOT", ROOT.parent / "vllm"))
POOL_SCHEDULER = ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = load_source("pool_load_link_diag", VLLM_ROOT / "vllm/v1/metrics/pd_diagnostics.py")
TREE = ast.parse(POOL_SCHEDULER.read_text())


def scheduler_method(name):
    for cls in TREE.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "KVPoolScheduler":
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return node
    raise AssertionError(name + " not found")


def compiled(name):
    node = scheduler_method(name)
    node.args.args = [a for a in node.args.args]
    for a in node.args.args:
        a.annotation = None
    node.returns = None
    namespace = {"diag": diag}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(POOL_SCHEDULER), "exec"), namespace)
    return namespace[name]


EMIT = compiled("_emit_load_links")


def scheduler(load_async=True, use_layerwise=False):
    return SimpleNamespace(
        load_async=load_async,
        use_layerwise=use_layerwise,
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(engine_id="0"),
            parallel_config=SimpleNamespace(data_parallel_rank=3),
        )
    )


def req(req_id, can_load, vllm=0, pool=256):
    spec = None if can_load is None else SimpleNamespace(can_load=can_load, vllm_cached_tokens=vllm, kvpool_cached_tokens=pool)
    return SimpleNamespace(req_id=req_id, load_spec=spec)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("[PD_DIAG] "):
            self.records.append(json.loads(message[len("[PD_DIAG] ") :]))


class PoolLoadLinkTest(unittest.TestCase):
    def setUp(self):
        self.capture = Capture()
        logger = logging.getLogger(diag.__name__)
        logger.addHandler(self.capture)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    def tearDown(self):
        logging.getLogger(diag.__name__).removeHandler(self.capture)

    def test_one_record_per_submitted_load(self):
        meta = SimpleNamespace(requests=[req("A", True, 16, 256), req("B", False), req("C", None), req("D", True)])
        output = SimpleNamespace(pd_execution_id="execution-r-11-5")
        EMIT(scheduler(), meta, output)
        self.assertEqual([r["request_id"] for r in self.capture.records], ["A", "D"])
        first = self.capture.records[0]
        for key, value in {
            "event": "pd_transfer_summary",
            "scope": "link",
            "transfer_role": "pool_load",
            "execution_id": "execution-r-11-5",
            "engine_id": "0",
            "dp_rank": 3,
            "load_async": True,
            "vllm_cached_tokens": 16,
            "kvpool_cached_tokens": 256,
            "status": "linked",
            "complete": True,
            "incomplete": False,
        }.items():
            self.assertEqual(first[key], value, key)

    def test_schedule_context_supplies_the_same_fields_as_the_consumer_link(self):
        meta = SimpleNamespace(requests=[req("A", True)])
        with diag.scope(execution_id="execution-r-11-6", request_ids=[], kv_role="kv_producer", role="unknown"):
            EMIT(scheduler(), meta, SimpleNamespace(pd_execution_id="execution-r-11-6"))
        (record,) = self.capture.records
        self.assertEqual(record["execution_id"], "execution-r-11-6")
        self.assertEqual(record["request_id"], "A")
        self.assertEqual(record["links"], [])

    def test_failure_never_reaches_the_scheduler(self):
        broken = SimpleNamespace(vllm_config=SimpleNamespace())
        EMIT(broken, SimpleNamespace(requests=[req("A", True)]), SimpleNamespace(pd_execution_id="x"))
        self.assertEqual(self.capture.records, [])

    def test_only_asynchronous_non_layerwise_loads_are_linked(self):
        meta = SimpleNamespace(requests=[req("A", True)])
        output = SimpleNamespace(pd_execution_id="x")
        EMIT(scheduler(load_async=False), meta, output)
        EMIT(scheduler(use_layerwise=True), meta, output)
        self.assertEqual(self.capture.records, [])
        EMIT(scheduler(), meta, output)
        self.assertEqual(len(self.capture.records), 1)

    def test_fixture_without_attributes_is_harmless(self):
        EMIT(SimpleNamespace(), SimpleNamespace(requests=[req("A", True)]), SimpleNamespace(pd_execution_id="x"))
        self.assertEqual(self.capture.records, [])

    def test_called_once_at_the_end_of_build_connector_meta(self):
        body = scheduler_method("build_connector_meta")
        calls = [n for n in ast.walk(body) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "_emit_load_links"]
        self.assertEqual(len(calls), 1)
        last = body.body[-2]
        self.assertIsInstance(last, ast.Expr)
        self.assertEqual(getattr(last.value.func, "attr", None), "_emit_load_links")

if __name__ == "__main__":
    unittest.main()
