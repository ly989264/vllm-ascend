"""Stdlib-only tests for the granularity of the Pool task records (ascend_store/diagnostics.py).

Three levels. The full one records everything. One below it leaves out the native call list, keeping the counters
that list feeds, so the task's own duration, its counters and its status read the same. The lowest one also leaves
out the stages that reach the record -- build_key, lookup, prepare, sync, put -- and does not time them; the
stages beside the record (associated_stages) and everything else are untouched at every level.

The vLLM side is a fake module in sys.modules, installed before the module under test is loaded by path, because
it imports pd_diagnostics at import time.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

# What the vLLM side answers with: the level, and where it came from (which is for the line it prints, not here).
DiagLevel = namedtuple("DiagLevel", "level source")

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/diagnostics.py"
LEVELS = (1, 2, 3)
RECORDS: list[tuple[str, dict]] = []
_level = [3]
_bare = [False]  # answer with the number itself instead of (level, source)


def _install_fake_vllm():
    @contextmanager
    def scope(**fields):
        yield fields

    diag = types.ModuleType("vllm.v1.metrics.pd_diagnostics")
    diag.scope = scope
    diag.current = lambda: {}
    diag.new_id = lambda prefix: prefix + "-1"
    diag.emit = lambda event, **fields: RECORDS.append((event, fields))
    diag.diag_level = lambda: _level[0] if _bare[0] else DiagLevel(_level[0], "env")
    metrics = types.ModuleType("vllm.v1.metrics")
    metrics.pd_diagnostics = diag
    v1 = types.ModuleType("vllm.v1")
    v1.metrics = metrics
    vllm = types.ModuleType("vllm")
    vllm.v1 = v1
    sys.modules.update({"vllm": vllm, "vllm.v1": v1, "vllm.v1.metrics": metrics,
                        "vllm.v1.metrics.pd_diagnostics": diag})


_install_fake_vllm()
spec = importlib.util.spec_from_file_location("pool_diagnostics_levels_ut", MODULE)
pd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pd  # the dataclass in it looks itself up there while the module runs
spec.loader.exec_module(pd)


def one_save_task(level):
    """One save task that exercises every kind of stage and one native call. Returns its record."""
    _level[0] = level
    pd._level_read.clear()
    del RECORDS[:]
    snapshot = pd.capture(None, "save")
    with pd.processing(snapshot):
        with pd.stage("build_key"):
            pass
        with pd.stage("sync"):
            pass
        with pd.stage("save_wait"):  # not one of the stages that reach the record
            pass
        with pd.backend("batch_put", 2, [[16], [32]], mode="ranges") as call:
            call.batch_result(0)
    tasks = [fields for event, fields in RECORDS if event == "pool_task"]
    assert len(tasks) == 1, tasks
    return tasks[0]


class LevelReadTest(unittest.TestCase):
    def setUp(self):
        pd._level_read.clear()
        self.addCleanup(pd._level_read.clear)
        self.addCleanup(lambda: _level.__setitem__(0, 3))

    def test_each_legal_level_is_taken_as_it_is(self):
        for level in LEVELS:
            pd._level_read.clear()
            _level[0] = level
            self.assertEqual(pd.level(), level)

    def test_a_bare_level_is_taken_too(self):
        """Should the vLLM side ever answer with the number itself."""
        _bare[0] = True
        self.addCleanup(lambda: _bare.__setitem__(0, False))
        for level in LEVELS:
            pd._level_read.clear()
            _level[0] = level
            self.assertEqual(pd.level(), level)

    def test_anything_else_is_the_full_level(self):
        for value in (0, 4, -1, "2", 2.0, None, True, False):
            pd._level_read.clear()
            _level[0] = value
            self.assertEqual(pd.level(), pd.FULL_LEVEL, f"{value!r} should not be taken as a level")

    def test_a_vllm_build_without_the_function_runs_at_the_full_level(self):
        diag = sys.modules["vllm.v1.metrics.pd_diagnostics"]
        saved = diag.diag_level
        del diag.diag_level
        self.addCleanup(lambda: setattr(diag, "diag_level", saved))
        self.assertEqual(pd.level(), pd.FULL_LEVEL)

    def test_a_raising_function_is_the_full_level(self):
        diag = sys.modules["vllm.v1.metrics.pd_diagnostics"]
        saved = diag.diag_level

        def boom():
            raise RuntimeError("sentinel")

        diag.diag_level = boom
        self.addCleanup(lambda: setattr(diag, "diag_level", saved))
        self.assertEqual(pd.level(), pd.FULL_LEVEL)

    def test_it_is_read_once_per_process(self):
        pd._level_read.clear()
        _level[0] = 1
        self.assertEqual(pd.level(), 1)
        _level[0] = 3  # a later change is not picked up: the value is settled at the first use
        self.assertEqual(pd.level(), 1)


class RecordTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(pd._level_read.clear)
        self.addCleanup(lambda: _level.__setitem__(0, 3))

    def test_the_full_level_records_the_stages_and_the_native_call(self):
        record = one_save_task(3)
        self.assertEqual(sorted(record["stages"]), ["build_key", "lookup", "prepare", "put", "sync"])
        self.assertEqual(record["stages"]["build_key"]["calls"], 1)
        self.assertEqual(record["stages"]["lookup"]["calls"], 0)  # filled in, never ran
        self.assertEqual(len(record["native_calls"]), 1)
        self.assertEqual(record["native_calls"][0]["operation"], "batch_put")
        self.assertEqual(record["native_call_count"], 1)

    def test_one_level_down_drops_only_the_native_call_list(self):
        record = one_save_task(2)
        self.assertEqual(sorted(record["stages"]), ["build_key", "lookup", "prepare", "put", "sync"])
        self.assertEqual(record["stages"]["build_key"]["calls"], 1)
        self.assertEqual(record["native_calls"], [])
        self.assertEqual(record["native_call_count"], 0)
        self.assertFalse(record["native_calls_dropped"])

    def test_the_lowest_level_also_drops_the_stages_that_reach_the_record(self):
        record = one_save_task(1)
        self.assertEqual(record["stages"], {})
        self.assertEqual(record["native_calls"], [])

    def test_what_does_not_move_with_the_level(self):
        full = one_save_task(3)
        for level in (1, 2):
            record = one_save_task(level)
            for field in ("scope", "operation", "status", "task_id", "member_count", "members",
                          "timing_scope", "timing_domain", "complete", "incomplete"):
                self.assertEqual(record.get(field), full.get(field), f"{field} at level {level}")
            # the clocks themselves differ from run to run; what matters is that they are still taken
            self.assertIsInstance(record["queue_enter_ns"], int, f"queue_enter_ns at level {level}")
            self.assertEqual(sorted(record), sorted(full), f"the record's fields at level {level}")
            # the counters the native call feeds, and the stages that sit beside the record
            self.assertEqual(sorted(record["counters"]), sorted(full["counters"]), f"counters at level {level}")
            for name, value in full["counters"].items():
                if not name.endswith("_ns"):  # a measured duration differs from run to run
                    self.assertEqual(record["counters"][name], value, f"counters.{name} at level {level}")
            self.assertEqual(sorted(record["associated_stages"]), ["backend_batch_put", "processing", "save_wait"],
                             f"associated_stages at level {level}")
            self.assertGreater(record["duration_ns"], 0)

    def test_the_lowest_level_does_not_time_those_stages_at_all(self):
        """Not just left out of the record: the clock is not read for them either."""
        _level[0] = 1
        pd._level_read.clear()
        reads = []
        saved = pd.time.monotonic_ns

        def counted():
            reads.append(1)
            return saved()

        pd.time.monotonic_ns = counted
        self.addCleanup(lambda: setattr(pd.time, "monotonic_ns", saved))
        snapshot = pd.capture(None, "save")
        with pd.processing(snapshot):
            before = len(reads)
            with pd.stage("build_key"):
                pass
            self.assertEqual(len(reads), before, "a stage that is not recorded reads no clock")
            with pd.stage("save_wait"):
                pass
            self.assertGreater(len(reads), before, "a stage beside the record is still timed")


if __name__ == "__main__":
    unittest.main()
