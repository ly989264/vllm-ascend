"""Stdlib-only tests for the granularity of the device stages in vllm_ascend/worker/pd_device_timing.py.

The granularity decides which measured stages reach the record, never which ones are measured. Only bookkeeping is
affected: below the full level its stage value is left out, while both of its device events are still recorded, so
the step envelope and the number of events on the card read the same at every level. The startup line does not
move with the level either. The device events are stubs on a scripted timeline and the vLLM side is a fake module
in sys.modules, as in the other tests of this module; this file carries its own copy so that it stands alone.
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

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("pd_device_timing_levels_ut",
                                              ROOT / "vllm_ascend/worker/pd_device_timing.py")
pdt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdt)
MS = 1_000_000
STREAM = object()
LEVELS = (1, 2, 3)
# The order the stages run in on a decode node: draft runs after bookkeeping there, so it closes the envelope.
STEP_ORDER = (
    ("execute", "prepare_inputs"),
    ("execute", "forward"),
    ("execute", "lm_head"),
    ("sample", "sampling"),
    ("sample", "bookkeeping"),
    ("sample", "draft"),
)
EVERY_STAGE = {name: 2 for _, name in STEP_ORDER}


class Device:
    def __init__(self):
        self.now_ms = 0.0
        self.completed = -1
        self.count = 0

    def complete_all(self):
        self.completed = self.count

    def event(self):
        return Event(self)


class Event:
    def __init__(self, device):
        self.device = device
        self.index = None
        self.ts = None

    def record(self, stream):
        assert stream is STREAM
        self.device.count += 1
        self.index = self.device.count
        self.ts = self.device.now_ms

    def query(self):
        return self.index is not None and self.index <= self.device.completed

    def elapsed_time(self, other):
        return other.ts - self.ts


class Batch:
    def __init__(self):
        self.fields: dict = {}

    def update(self, **fields):
        self.fields.update(fields)


class Claim:
    def __init__(self):
        self.stages: dict = {}

    def set_stage_fields(self, stage, **fields):
        self.stages.setdefault(stage, {}).update(fields)
        return True

    def release(self):
        pass


class Side:
    """The claim and record side of one process, keyed by (execution id, phase)."""

    def __init__(self):
        self.context = {"execution_id": None, "phase": None}
        self.claims: dict[tuple, Claim] = {}
        self.batches: dict[tuple, Batch] = {}

    def key(self):
        return (self.context["execution_id"], self.context["phase"])

    def claim_finish(self):
        return None if self.context["execution_id"] is None else self.claims.setdefault(self.key(), Claim())

    def current_batch(self):
        return None if self.context["execution_id"] is None else self.batches.setdefault(self.key(), Batch())


def make(level, stages=pdt.OPTIONAL_STAGES):
    device, side = Device(), Side()
    timing = pdt.DeviceTiming(
        enabled=True,
        new_event=device.event,
        current_stream=lambda: STREAM,
        capturing=lambda: False,
        claim_finish=side.claim_finish,
        current_context=lambda: dict(side.context),
        current_batch=side.current_batch,
        stages=stages,
        decode_node=True,
        level=level,
    )
    return timing, device, side


def run_step(timing, device, side, execution_id="e1", durations=EVERY_STAGE):
    """One step in the order the runner measures, through measure() as the runner's hook does."""
    side.context.update(execution_id=execution_id, phase="execute")
    timing.begin_execute()
    for phase, name in STEP_ORDER:
        if phase == "sample" and side.context["phase"] != "sample":
            side.context["phase"] = "sample"
            timing.begin_sample()
        if name in pdt.OPTIONAL_STAGES and name not in timing.stages:
            device.now_ms += durations.get(name, 0)
            continue
        with timing.measure(name):
            device.now_ms += durations.get(name, 0)
    device.complete_all()
    timing.end_sample(read_now=True)


def install_fake_vllm():
    """The vLLM names the hook imports at call time. Returns (the record it writes to, span names opened, undo)."""
    names = ("vllm", "vllm.v1", "vllm.v1.metrics", "vllm.v1.metrics.pd_diagnostics")
    saved = {name: sys.modules.get(name) for name in names}
    opened: list[str] = []
    record = Batch()

    @contextmanager
    def span(name, **fields):
        opened.append(name)
        yield {}

    diag = types.ModuleType("vllm.v1.metrics.pd_diagnostics")
    diag.span = span
    diag.current_batch = lambda: record
    metrics = types.ModuleType("vllm.v1.metrics")
    metrics.pd_diagnostics = diag
    v1 = types.ModuleType("vllm.v1")
    v1.metrics = metrics
    vllm = types.ModuleType("vllm")
    vllm.v1 = v1
    sys.modules.update({"vllm": vllm, "vllm.v1": v1, "vllm.v1.metrics": metrics,
                        "vllm.v1.metrics.pd_diagnostics": diag})

    def undo():
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    return record, opened, undo


class Runner:
    def __init__(self, timing):
        self._pd_device_timing = timing


class LevelReadTest(unittest.TestCase):
    """_diag_level(): the one place the vLLM side's answer becomes a level."""

    @staticmethod
    def diag(**kwargs):
        return types.SimpleNamespace(**kwargs)

    def test_each_legal_level_is_taken_as_it_is(self):
        for level in LEVELS:
            answer = DiagLevel(level, "env")
            self.assertEqual(pdt._diag_level(self.diag(diag_level=lambda a=answer: a)), level)

    def test_a_bare_level_is_taken_too(self):
        """Should the vLLM side ever answer with the number itself."""
        for level in LEVELS:
            self.assertEqual(pdt._diag_level(self.diag(diag_level=lambda level=level: level)), level)

    def test_the_source_beside_the_level_is_ignored(self):
        for source in ("default", "env", "invalid"):
            self.assertEqual(pdt._diag_level(self.diag(diag_level=lambda s=source: DiagLevel(1, s))), 1)

    def test_a_vllm_patch_without_the_function_runs_at_the_full_level(self):
        self.assertEqual(pdt._diag_level(self.diag()), pdt.FULL_LEVEL)
        self.assertEqual(pdt._diag_level(None), pdt.FULL_LEVEL)

    def test_anything_that_is_not_one_of_the_three_levels_is_the_full_level(self):
        values = (0, 4, -1, "2", 2.0, None, True, False, object(), (2, "env"),
                  DiagLevel(0, "env"), DiagLevel("2", "env"), DiagLevel(None, "env"))
        for value in values:
            self.assertEqual(pdt._diag_level(self.diag(diag_level=lambda value=value: value)), pdt.FULL_LEVEL,
                             f"{value!r} should not be taken as a level")

    def test_a_raising_function_is_the_full_level(self):
        def boom():
            raise RuntimeError("sentinel")

        self.assertEqual(pdt._diag_level(self.diag(diag_level=boom)), pdt.FULL_LEVEL)


class StartupLineTest(unittest.TestCase):
    def test_the_line_reads_the_same_at_every_level(self):
        for level in LEVELS:
            timing, _, _ = make(level)
            self.assertEqual(timing.runner_stages,
                             ("forward", "sampling", "prepare_inputs", "lm_head", "draft", "bookkeeping"),
                             f"level {level}")
            self.assertEqual(timing.stages_source, "default")
            self.assertIn("bookkeeping", timing.stages, f"bookkeeping is measured at level {level}")


class ReportsTest(unittest.TestCase):
    def test_only_bookkeeping_is_held_back_and_only_below_the_full_level(self):
        for level in LEVELS:
            timing, _, _ = make(level)
            for name in ("forward", "sampling", "prepare_inputs", "lm_head", "draft"):
                self.assertTrue(timing.reports(name), f"{name} at level {level}")
            self.assertEqual(timing.reports("bookkeeping"), level >= pdt.FULL_LEVEL, f"bookkeeping at level {level}")


class RecordTest(unittest.TestCase):
    def step(self, level, stages=pdt.OPTIONAL_STAGES):
        timing, device, side = make(level, stages=stages)
        run_step(timing, device, side)
        return device, side

    @staticmethod
    def values(side, phase):
        return sorted(side.claims[("e1", phase)].stages)

    def test_the_execute_record_does_not_move_with_the_level(self):
        for level in LEVELS:
            _, side = self.step(level)
            self.assertEqual(self.values(side, "execute"), ["forward", "lm_head", "prepare_inputs"], f"level {level}")

    def test_the_bookkeeping_value_is_written_only_at_the_full_level(self):
        for level in LEVELS:
            _, side = self.step(level)
            expected = ["bookkeeping", "draft", "sampling"] if level == pdt.FULL_LEVEL else ["draft", "sampling"]
            self.assertEqual(self.values(side, "sample"), expected, f"level {level}")

    def test_the_envelope_and_the_events_on_the_card_are_the_same_at_every_level(self):
        spans, counts = set(), set()
        for level in LEVELS:
            device, side = self.step(level)
            spans.add(side.batches[("e1", "sample")].fields["step_device_span_ns"])
            counts.add(device.count)
        # Six stages of 2 ms measured end to end, and two events per stage, whatever the level.
        self.assertEqual(spans, {12 * MS})
        self.assertEqual(counts, {2 * len(STEP_ORDER)})

    def test_bookkeeping_switched_off_leaves_nothing_of_it_at_any_level(self):
        for level in LEVELS:
            device, side = self.step(level, stages=("prepare_inputs", "lm_head", "draft"))
            self.assertNotIn("bookkeeping", self.values(side, "sample"), f"level {level}")
            self.assertEqual(device.count, 2 * (len(STEP_ORDER) - 1), f"level {level}")


class HelperTest(unittest.TestCase):
    """Through the hook the runner calls, which is where the host span is opened."""

    def one_stage(self, name, level):
        """Drives one stage through the hook, in the phase that stage belongs to."""
        phase = pdt._PHASE_OF_STAGE[name]
        timing, device, side = make(level)
        _, opened, undo = install_fake_vllm()
        self.addCleanup(undo)
        side.context.update(execution_id="e1", phase="execute")
        timing.begin_execute()
        if phase == "sample":
            side.context["phase"] = "sample"
            timing.begin_sample()
        with pdt.stage(Runner(timing), name):
            device.now_ms += 3
        device.complete_all()
        timing.end_sample(read_now=True)
        claim = side.claims.get(("e1", phase))
        return opened, device, {} if claim is None else claim.stages

    def test_a_reported_stage_opens_its_span_at_every_level(self):
        for level in LEVELS:
            opened, _, stages = self.one_stage("lm_head", level)
            self.assertEqual(opened, ["runner.lm_head"], f"level {level}")
            self.assertEqual(stages["lm_head"]["device_duration_ns"], 3 * MS)

    def test_bookkeeping_opens_its_span_only_at_the_full_level(self):
        opened, device, stages = self.one_stage("bookkeeping", pdt.FULL_LEVEL)
        self.assertEqual(opened, ["runner.bookkeeping"])
        self.assertEqual(device.count, 2)
        self.assertEqual(stages["bookkeeping"]["device_duration_ns"], 3 * MS)

    def test_below_the_full_level_the_events_are_recorded_and_nothing_is_written(self):
        for level in (1, 2):
            opened, device, stages = self.one_stage("bookkeeping", level)
            self.assertEqual(opened, [], f"no host span at level {level}")
            self.assertEqual(device.count, 2, f"both device events at level {level}")
            self.assertEqual(stages, {}, f"no stage value at level {level}")

    def test_the_body_still_runs_when_the_stage_is_not_reported(self):
        timing, _, _ = make(1)
        ran = False
        with pdt.stage(Runner(timing), "bookkeeping"):
            ran = True
        self.assertTrue(ran)

    def test_the_body_error_still_reaches_the_caller(self):
        timing, _, _ = make(1)
        with self.assertRaises(ValueError):
            with pdt.stage(Runner(timing), "bookkeeping"):
                raise ValueError("sentinel")


class CudagraphModeTest(unittest.TestCase):
    def test_the_mode_is_written_at_every_level(self):
        for level in LEVELS:
            timing, _, _ = make(level)
            record, _, undo = install_fake_vllm()
            self.addCleanup(undo)
            pdt.note_cudagraph_mode(Runner(timing), types.SimpleNamespace(runtime_mode="FULL"), None)
            self.assertEqual(record.fields, {"cudagraph_runtime_mode": "FULL"}, f"level {level}")


if __name__ == "__main__":
    unittest.main()
