"""Stdlib-only tests for the optional decode-node stages of vllm_ascend/worker/pd_device_timing.py.

Two stages per step are always measured; four more are optional and switchable, so that a prefill node -- and a
decode node with the switch off -- writes exactly what those two alone wrote. The device events are stubs on a
scripted timeline, as in test_pd_device_timing.py; the vLLM side (pd_diagnostics, forward context) is a fake module
installed into sys.modules, because the helper imports it at call time.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("pd_device_timing_stages_ut", ROOT / "vllm_ascend/worker/pd_device_timing.py")
pdt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdt)
MS = 1_000_000
STREAM = object()
# The order the stages actually run in on a decode node (draft last: it runs after bookkeeping there).
STEP_ORDER = (
    ("execute", "prepare_inputs"),
    ("execute", "forward"),
    ("execute", "lm_head"),
    ("sample", "sampling"),
    ("sample", "bookkeeping"),
    ("sample", "draft"),
)


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
    """vLLM's diag.Batch as far as this module uses it: record-level fields."""

    def __init__(self):
        self.fields: dict = {}

    def update(self, **fields):
        self.fields.update(fields)


class Claim:
    def __init__(self):
        self.stages: dict = {}
        self.released = 0

    def set_stage_fields(self, stage, **fields):
        self.stages.setdefault(stage, {}).update(fields)
        return True

    def release(self):
        self.released += 1


class Side:
    """The claim and record side of one process, keyed by phase."""

    def __init__(self):
        self.context = {"execution_id": None, "phase": None}
        self.claims: dict[tuple, Claim] = {}
        self.batches: dict[tuple, Batch] = {}

    def key(self):
        return (self.context["execution_id"], self.context["phase"])

    def claim_finish(self):
        if self.context["execution_id"] is None:
            return None
        return self.claims.setdefault(self.key(), Claim())

    def current_batch(self):
        if self.context["execution_id"] is None:
            return None
        return self.batches.setdefault(self.key(), Batch())


def make(stages=pdt.OPTIONAL_STAGES, decode_node=True):
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
        decode_node=decode_node,
    )
    return timing, device, side


def run_step(timing, device, side, execution_id, durations, synchronous=True):
    """One step in the order the runner measures: the four optional stages only when they are on."""
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
    if synchronous:
        device.complete_all()
    timing.end_sample(read_now=synchronous)


def install_fake_vllm(spans=True, is_decode=True, batch=None):
    """The three vLLM names this module imports at call time. Returns (batch, calls, undo)."""
    saved = {name: sys.modules.get(name) for name in
             ("vllm", "vllm.v1", "vllm.v1.metrics", "vllm.v1.metrics.pd_diagnostics")}
    calls: list[str] = []
    record = batch if batch is not None else Batch()

    @contextmanager
    def span(name, **fields):
        calls.append(name)
        yield {}

    diag = types.ModuleType("vllm.v1.metrics.pd_diagnostics")
    diag.span = span
    diag.current_batch = lambda: record
    diag._SPANS = {"runner." + name: name for name in pdt.OPTIONAL_STAGES} if spans else {}
    if is_decode is not None:
        diag.is_pd_decode = lambda vllm_config=None: is_decode
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

    return record, calls, undo


class Runner:
    """What the hooks read off the model runner."""

    def __init__(self, timing):
        self._pd_device_timing = timing


class SwitchTest(unittest.TestCase):
    def test_unset_means_all_four_stages(self):
        self.assertEqual(pdt._parse_stages(None), (pdt.OPTIONAL_STAGES, "default"))

    def test_none_goes_back_to_the_base_behaviour(self):
        self.assertEqual(pdt._parse_stages("none"), ((), "env"))
        self.assertEqual(pdt._parse_stages("NONE"), ((), "env"))
        self.assertEqual(pdt._parse_stages(" none , lm_head "), ((), "env"))

    def test_a_subset_keeps_the_step_order(self):
        self.assertEqual(pdt._parse_stages("draft,lm_head"), (("lm_head", "draft"), "env"))
        self.assertEqual(pdt._parse_stages(" bookkeeping , prepare_inputs "),
                         (("prepare_inputs", "bookkeeping"), "env"))

    def test_unknown_names_are_ignored_and_the_base_stages_are_not_switchable(self):
        self.assertEqual(pdt._parse_stages("forward,sampling"), ((), "env"))
        self.assertEqual(pdt._parse_stages("lm_head,typo"), (("lm_head",), "env"))
        self.assertEqual(pdt._parse_stages(""), ((), "env"))

    def test_runner_stages_tells_switched_off_apart_from_not_a_decode_node(self):
        timing, _, _ = make(stages=("lm_head", "draft"))
        self.assertEqual(timing.runner_stages, ("forward", "sampling", "lm_head", "draft"))
        self.assertEqual(timing.stages_source, "default")
        # Switched off on a decode node: the line still says so.
        off, _, _ = make(stages=())
        self.assertEqual(off.runner_stages, ("forward", "sampling"))
        # Not a decode node (a prefill node, or no way to tell): nothing on the line, as in the base behaviour.
        not_decode, _, _ = make(stages=pdt.OPTIONAL_STAGES, decode_node=False)
        self.assertEqual(not_decode.runner_stages, ())
        self.assertEqual(not_decode.stages, frozenset())
        self.assertEqual(pdt._DISABLED.runner_stages, ())
        self.assertEqual(pdt._DISABLED.stages, frozenset())

    def test_a_disabled_timing_measures_nothing(self):
        timing = pdt.DeviceTiming(enabled=False, stages=pdt.OPTIONAL_STAGES, decode_node=True)
        self.assertEqual(timing.stages, frozenset())
        self.assertFalse(timing.decode_node)


class DecodeNodeTest(unittest.TestCase):
    def fake(self, **kwargs):
        record, calls, undo = install_fake_vllm(**kwargs)
        self.addCleanup(undo)
        return sys.modules["vllm.v1.metrics.pd_diagnostics"], record, calls

    def test_decode_node_with_the_spans_registered(self):
        diag, _, _ = self.fake()
        self.assertTrue(pdt._is_decode_node(diag, object()))

    def test_prefill_node(self):
        diag, _, _ = self.fake(is_decode=False)
        self.assertFalse(pdt._is_decode_node(diag, object()))

    def test_an_older_vllm_patch_without_the_function_or_the_spans(self):
        diag, _, _ = self.fake(is_decode=None)
        self.assertFalse(pdt._is_decode_node(diag, object()))
        diag, _, _ = self.fake(spans=False)
        self.assertFalse(pdt._is_decode_node(diag, object()))

    def test_a_raising_function_is_a_no(self):
        diag, _, _ = self.fake()

        def boom(vllm_config=None):
            raise RuntimeError("sentinel")

        diag.is_pd_decode = boom
        self.assertFalse(pdt._is_decode_node(diag, object()))


class SwitchGateTest(unittest.TestCase):
    """DeviceTiming.switches(): the gate and the environment together, the way for_runner uses them."""

    def setUp(self):
        os.environ.pop(pdt._ENV_STAGES, None)
        self.addCleanup(lambda: os.environ.pop(pdt._ENV_STAGES, None))

    def fake(self, **kwargs):
        _, _, undo = install_fake_vllm(**kwargs)
        self.addCleanup(undo)
        return sys.modules["vllm.v1.metrics.pd_diagnostics"]

    def test_decode_node_defaults_to_all_four(self):
        self.assertEqual(pdt.DeviceTiming.switches(self.fake(), None), (pdt.OPTIONAL_STAGES, "default", True))

    def test_decode_node_with_the_switch_set(self):
        os.environ[pdt._ENV_STAGES] = "draft"
        self.assertEqual(pdt.DeviceTiming.switches(self.fake(), None), (("draft",), "env", True))

    def test_prefill_node_measures_nothing_whatever_the_switch_says(self):
        os.environ[pdt._ENV_STAGES] = "prepare_inputs,lm_head,draft,bookkeeping"
        stages, _, decode = pdt.DeviceTiming.switches(self.fake(is_decode=False), None)
        self.assertEqual((stages, decode), ((), False))

    def test_an_older_vllm_patch_measures_nothing(self):
        stages, _, decode = pdt.DeviceTiming.switches(self.fake(spans=False), None)
        self.assertEqual((stages, decode), ((), False))


class SixStageTest(unittest.TestCase):
    def values(self, side, execution_id, phase):
        return side.claims[(execution_id, phase)].stages

    def test_each_stage_goes_to_its_own_record(self):
        timing, device, side = make()
        run_step(timing, device, side, "e1", {name: 1 for _, name in STEP_ORDER})
        execute = self.values(side, "e1", "execute")
        sample = self.values(side, "e1", "sample")
        self.assertEqual(sorted(execute), ["forward", "lm_head", "prepare_inputs"])
        self.assertEqual(sorted(sample), ["bookkeeping", "draft", "sampling"])
        for stages in (execute, sample):
            for fields in stages.values():
                self.assertEqual(fields["device_duration_ns"], 1 * MS)

    def test_the_step_envelope_spans_the_first_stage_to_the_last(self):
        timing, device, side = make()
        run_step(timing, device, side, "e1", {name: 2 for _, name in STEP_ORDER})
        # Six stages of 2 ms each, measured end to end.
        self.assertEqual(side.batches[("e1", "sample")].fields["step_device_span_ns"], 12 * MS)
        self.assertNotIn("step_device_span_ns", side.batches[("e1", "execute")].fields)

    def test_no_envelope_when_either_end_of_it_is_switched_off(self):
        for stages in (("lm_head", "draft"), ("prepare_inputs", "lm_head")):
            timing, device, side = make(stages=stages)
            run_step(timing, device, side, "e1", {name: 2 for _, name in STEP_ORDER})
            self.assertNotIn("step_device_span_ns", side.batches[("e1", "sample")].fields)

    def test_a_switched_off_stage_is_not_in_the_record_at_all(self):
        timing, device, side = make(stages=("lm_head",))
        run_step(timing, device, side, "e1", {name: 1 for _, name in STEP_ORDER})
        self.assertEqual(sorted(self.values(side, "e1", "execute")), ["forward", "lm_head"])
        self.assertEqual(sorted(self.values(side, "e1", "sample")), ["sampling"])

    def test_with_every_stage_off_the_records_are_what_the_base_wrote(self):
        timing, device, side = make(stages=())
        self.assertEqual(timing.runner_stages, ("forward", "sampling"))  # the line still marks it a decode node
        run_step(timing, device, side, "e1", {name: 1 for _, name in STEP_ORDER})
        self.assertEqual(sorted(self.values(side, "e1", "execute")), ["forward"])
        self.assertEqual(sorted(self.values(side, "e1", "sample")), ["sampling"])
        for batch in side.batches.values():
            self.assertEqual(batch.fields, {})

    def test_events_that_never_complete_leave_the_stages_without_device_values(self):
        timing, device, side = make()
        run_step(timing, device, side, "e1", {name: 1 for _, name in STEP_ORDER}, synchronous=False)
        for _ in range(pdt.SLOTS):
            side.context.update(execution_id=None, phase=None)
            timing.begin_execute()
        self.assertEqual(self.values(side, "e1", "execute"), {})
        self.assertEqual(side.batches[("e1", "sample")].fields, {})


class HelperTest(unittest.TestCase):
    def test_the_helper_opens_the_span_and_the_events_together(self):
        timing, device, side = make()
        _, calls, undo = install_fake_vllm()
        self.addCleanup(undo)
        side.context.update(execution_id="e1", phase="execute")
        timing.begin_execute()
        with pdt.stage(Runner(timing), "lm_head"):
            device.now_ms += 3
        device.complete_all()
        timing.end_sample(read_now=True)
        self.assertEqual(calls, ["runner.lm_head"])
        self.assertEqual(side.claims[("e1", "execute")].stages["lm_head"]["device_duration_ns"], 3 * MS)

    def test_a_stage_that_is_off_opens_nothing(self):
        timing, device, side = make(stages=("draft",))
        _, calls, undo = install_fake_vllm()
        self.addCleanup(undo)
        side.context.update(execution_id="e1", phase="execute")
        timing.begin_execute()
        ran = False
        with pdt.stage(Runner(timing), "lm_head"):
            ran = True
        self.assertTrue(ran)
        self.assertEqual(calls, [])
        device.complete_all()
        timing.end_sample(read_now=True)
        self.assertEqual(side.claims[("e1", "execute")].stages, {})

    def test_the_helper_never_raises_without_the_vllm_module(self):
        timing, _, _ = make()
        ran = False
        with pdt.stage(Runner(timing), "draft"):
            ran = True
        self.assertTrue(ran)

    def test_the_body_error_reaches_the_caller(self):
        timing, _, _ = make()
        _, _, undo = install_fake_vllm()
        self.addCleanup(undo)
        with self.assertRaises(ValueError):
            with pdt.stage(Runner(timing), "draft"):
                raise ValueError("sentinel")


class CudagraphModeTest(unittest.TestCase):
    def test_the_mode_is_written_on_the_execute_record(self):
        timing, _, _ = make()
        record, _, undo = install_fake_vllm()
        self.addCleanup(undo)
        pdt.note_cudagraph_mode(Runner(timing), types.SimpleNamespace(runtime_mode="FULL"), None)
        self.assertEqual(record.fields, {"cudagraph_runtime_mode": "FULL"})

    def test_the_enum_name_is_used_when_there_are_no_stats(self):
        timing, _, _ = make()
        record, _, undo = install_fake_vllm()
        self.addCleanup(undo)
        pdt.note_cudagraph_mode(Runner(timing), None, types.SimpleNamespace(name="PIECEWISE"))
        self.assertEqual(record.fields, {"cudagraph_runtime_mode": "PIECEWISE"})

    def test_nothing_is_written_when_every_stage_is_off(self):
        timing, _, _ = make(stages=())
        record, _, undo = install_fake_vllm()
        self.addCleanup(undo)
        pdt.note_cudagraph_mode(Runner(timing), types.SimpleNamespace(runtime_mode="FULL"), None)
        self.assertEqual(record.fields, {})

    def test_nothing_is_written_and_nothing_raises_without_a_mode(self):
        timing, _, _ = make()
        record, _, undo = install_fake_vllm()
        self.addCleanup(undo)
        pdt.note_cudagraph_mode(Runner(timing), None, None)
        pdt.note_cudagraph_mode(Runner(timing))
        self.assertEqual(record.fields, {})


if __name__ == "__main__":
    unittest.main()
