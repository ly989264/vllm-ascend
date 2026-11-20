"""Stdlib-only tests for the runner's forward and sampling measurement points (d4ad49796 layout).

Set VLLM_COMPACT_ROOT to the vLLM checkout that carries vllm/v1/metrics/pd_diagnostics.py.
The measured methods are compiled from the runner sources, so neither torch nor NPU packages load;
the model, samplers and helpers they call are fakes that record when they ran. Static checks cover
where the execute/sample decorators and the measured calls sit.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import json
import logging
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VLLM_ROOT = Path(os.environ.get("VLLM_COMPACT_ROOT", ROOT.parent / "vllm"))
V1 = ROOT / "vllm_ascend/worker/model_runner_v1.py"
V2 = ROOT / "vllm_ascend/worker/v2/model_runner.py"
MS = 1_000_000


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = load_source("runner_points_diag", VLLM_ROOT / "vllm/v1/metrics/pd_diagnostics.py")
pdt = load_source("runner_points_device_timing", ROOT / "vllm_ascend/worker/pd_device_timing.py")


def runner_methods(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner":
            return {f.name: f for f in node.body if isinstance(f, ast.FunctionDef)}
    raise AssertionError(f"NPUModelRunner not found in {path}")


def compile_method(path, name, namespace):
    node = runner_methods(path)[name]
    node = ast.FunctionDef(**{**{f: getattr(node, f) for f in node._fields}, "decorator_list": []})
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class Clock:
    """Records the monotonic window of each fake call."""

    def __init__(self):
        self.calls = {}

    def timed(self, name, ms, result=None):
        def call(*args, **kwargs):
            start = time.monotonic_ns()
            time.sleep(ms / 1000)
            self.calls.setdefault(name, []).append((start, time.monotonic_ns()))
            return result

        return call


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        message = record.getMessage()
        if "[PD_DIAG] " in message:
            self.records.append(json.loads(message.split("[PD_DIAG] ", 1)[1]))


def run_in_step(function, phase="execute"):
    handler = Capture()
    diag._LOG.addHandler(handler)
    level = diag._LOG.level
    diag._LOG.setLevel(logging.INFO)
    try:
        context = {"execution_id": "exec-runner", "request_id": None, "request_ids": ["r0"], "tp_rank": 0}
        with diag.scope(**context), diag.batch("worker_summary", phase=phase, **context):
            function()
    finally:
        diag._LOG.removeHandler(handler)
        diag._LOG.setLevel(level)
    (record,) = [r for r in handler.records if r["event"] == "worker_summary"]
    return record


class RunnerForwardSamplingTest(unittest.TestCase):
    def assert_inside(self, stage, window):
        self.assertLessEqual(stage["start_ns"], window[0])
        self.assertLessEqual(window[1], stage["end_ns"])

    def assert_outside(self, stage, window):
        self.assertTrue(window[1] <= stage["start_ns"] or stage["end_ns"] <= window[0], (stage, window))

    def test_v1_forward_covers_only_the_target_model_call(self):
        tensor = type("Tensor", (), {})
        namespace = {
            "diag": diag,
            "pd_device_timing": pdt,
            "partial": functools.partial,
            "Any": Any,
            "IntermediateTensors": type("IntermediateTensors", (), {}),
            "torch": SimpleNamespace(Tensor=tensor, npu=SimpleNamespace(is_current_stream_capturing=lambda: False)),
            "CUDAGraphMode": SimpleNamespace(FULL="FULL"),
        }
        for enable_enpu in (False, True):
            with self.subTest(enable_enpu=enable_enpu):
                clock = Clock()
                context = SimpleNamespace(cudagraph_runtime_mode="NONE", device_metadata_executor=None, capturing=False)
                namespace["get_forward_context"] = lambda context=context: context
                forward = compile_method(V1, "_model_forward", dict(namespace))
                model = type("Model", (), {"__call__": staticmethod(clock.timed("model", 20, "hidden"))})()
                runner = SimpleNamespace(
                    model=model,
                    enable_enpu=enable_enpu,
                    _update_full_graph_params_if_needed=clock.timed("update_graph_params", 15),
                )
                result = []
                record = run_in_step(lambda: result.append(forward(runner, 8, input_ids="ids", positions="pos")))
                self.assertEqual(result, ["hidden"])
                stage = record["stages"]["forward"]
                self.assertEqual((stage["calls"], stage["status"], stage["completion"]), (1, "ok", "host_return"))
                self.assertGreaterEqual(stage["duration_ns"], 20 * MS)
                self.assert_inside(stage, clock.calls["model"][0])
                self.assert_outside(stage, clock.calls["update_graph_params"][0])

    def test_v1_sampling_covers_only_the_sampler_call(self):
        namespace = {"diag": diag, "lmhead_tp_enable": lambda: False, "pd_device_timing": pdt}
        sample = compile_method(V1, "_sample", namespace)
        for algorithm, spec in (("sampler", None), ("rejection", SimpleNamespace(logits_indices=[0, 1]))):
            with self.subTest(algorithm=algorithm):
                clock = Clock()
                runner = SimpleNamespace(
                    input_batch=SimpleNamespace(
                        update_async_output_token_ids=clock.timed("update_tokens", 10),
                        sampling_metadata="metadata",
                        num_reqs=1,
                    ),
                    sampler=clock.timed("sampler", 12, "sampled"),
                    rejection_sampler=clock.timed("rejection", 14, "accepted"),
                    _get_spec_decode_draft_probs=clock.timed("draft_probs", 16),
                )
                result = []
                record = run_in_step(lambda: result.append(sample(runner, "logits", spec)), phase="sample")
                self.assertEqual(result, ["sampled" if spec is None else "accepted"])
                stage = record["stages"]["sampling"]
                self.assertEqual((stage["calls"], stage["algorithm"], stage["completion"]), (1, algorithm, "host_return"))
                self.assert_inside(stage, clock.calls["sampler" if spec is None else "rejection"][0])
                self.assert_outside(stage, clock.calls["update_tokens"][0])
                if spec is not None:
                    self.assert_outside(stage, clock.calls["draft_probs"][0])

    def test_v2_lmhead_sampling_covers_only_the_sampler_call(self):
        namespace = {"diag": diag, "lmhead_tp_enable": lambda: True, "lmhead_tp_pad_rows": lambda rows, *args: rows}
        sample = compile_method(V2, "sample", namespace)
        for algorithm, drafts in (("sampler", 0), ("rejection", 2)):
            with self.subTest(algorithm=algorithm):
                clock = Clock()
                output = SimpleNamespace(num_sampled=1, num_rejected=0)
                runner = SimpleNamespace(
                    _lmhead_tp_max_num_logits=lambda: 4,
                    model=SimpleNamespace(compute_logits=clock.timed("compute_logits", 10, ["l0", "l1", "l2", "l3"])),
                    structured_outputs_worker=None,
                    sampler=clock.timed("sampler", 12, output),
                    rejection_sampler=clock.timed("rejection", 14, output),
                    speculator=SimpleNamespace(draft_logits="draft"),
                )
                batch = SimpleNamespace(logits_indices=SimpleNamespace(shape=[2]), num_draft_tokens=drafts)
                hidden_states = type("Hidden", (), {"__getitem__": lambda self, index: "h"})()
                record = run_in_step(lambda: sample(runner, hidden_states, batch, None), phase="sample")
                stage = record["stages"]["sampling"]
                self.assertEqual((stage["calls"], stage["algorithm"]), (1, algorithm))
                self.assert_inside(stage, clock.calls["sampler" if drafts == 0 else "rejection"][0])
                self.assert_outside(stage, clock.calls["compute_logits"][0])


def decorators(function):
    return [ast.unparse(node.func if isinstance(node, ast.Call) else node) for node in function.decorator_list]


def call_spans(function, call_text):
    found = []

    def visit(node, span):
        if isinstance(node, ast.With):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call) and ast.unparse(call.func) == "diag.span" and call.args:
                    span = ast.literal_eval(call.args[0])
        if isinstance(node, ast.Call) and ast.unparse(node.func) == call_text:
            found.append(span)
        for child in ast.iter_child_nodes(node):
            visit(child, span)

    visit(function, None)
    return found


def stage_blocks(function, name=None):
    """The `with pd_device_timing.stage(self, "<name>")` blocks inside this function, by stage name."""
    found = {}

    def visit(node):
        if isinstance(node, ast.With):
            for item in node.items:
                call = item.context_expr
                if (isinstance(call, ast.Call) and ast.unparse(call.func) == "pd_device_timing.stage"
                        and len(call.args) == 2):
                    found.setdefault(ast.literal_eval(call.args[1]), []).append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(function)
    return found if name is None else found.get(name, [])


def calls_in(node):
    return {ast.unparse(child.func) for child in ast.walk(node) if isinstance(child, ast.Call)}


class RunnerPlacementTest(unittest.TestCase):
    def test_v1_decorators_and_call_sites(self):
        v1 = runner_methods(V1)
        self.assertEqual(decorators(v1["execute_model"])[:2], ["torch.inference_mode", "pd_execution"])
        self.assertEqual(decorators(v1["sample_tokens"])[:2], ["torch.inference_mode", "pd_sample"])
        # The wait_for_put call chain starts inside execute_model, so it lands in phase=execute.
        self.assertEqual(len(call_spans(v1["execute_model"], "get_kv_transfer_group().handle_preemptions")), 1)
        self.assertEqual(call_spans(v1["_model_forward"], "run_model"), ["runner.forward", "runner.forward"])
        sampled = call_spans(v1["_sample"], "self.sampler") + call_spans(v1["_sample"], "self.rejection_sampler")
        self.assertEqual(sampled, ["runner.sampling", "runner.sampling"])
        self.assertEqual({n for n, f in v1.items() if call_spans(f, "self._sample")}, {"sample_tokens"})
        self.assertEqual({n for n, f in v1.items() if call_spans(f, "self._model_forward")}, {"execute_model", "_dummy_run"})
        # MTP draft proposal is outside the forward and sampling spans.
        drafts = call_spans(v1["sample_tokens"], "self.propose_draft_token_ids")
        self.assertTrue(drafts)
        self.assertTrue(all(span is None for span in drafts))

    def test_v2_decorators_and_call_sites(self):
        v2 = runner_methods(V2)
        self.assertIn("pd_execution", decorators(v2["execute_model"]))
        self.assertIn("pd_sample", decorators(v2["sample_tokens"]))
        self.assertEqual(len(call_spans(v2["execute_model"], "get_kv_transfer_group().handle_preemptions")), 1)
        sampled = call_spans(v2["sample"], "self.sampler") + call_spans(v2["sample"], "self.rejection_sampler")
        self.assertEqual(sampled, ["runner.sampling", "runner.sampling"])


    def test_v1_device_timing_hooks(self):
        v1 = runner_methods(V1)
        calls = lambda name, text: len(call_spans(v1[name], text))  # noqa: E731
        self.assertEqual(calls("execute_model", "pd_device_timing.begin_execute"), 1)
        self.assertEqual(calls("sample_tokens", "pd_device_timing.begin_sample"), 1)
        self.assertEqual(calls("sample_tokens", "pd_device_timing.end_sample"), 2)
        self.assertEqual(calls("_model_forward", "pd_device_timing.measure"), 2)
        self.assertEqual(calls("_sample", "pd_device_timing.measure"), 2)
        source = ast.unparse(v1["sample_tokens"])
        self.assertIn("pd_device_timing.async_output_class(AsyncGPUModelRunnerOutput)(", source)
        # Each end_sample call is the statement right before its return.
        self.assertRegex(source, r"pd_device_timing\.end_sample\(self, read_now=True\)\s+return model_runner_output")
        self.assertRegex(
            source, r"pd_device_timing\.end_sample\(self, read_now=False, async_output=async_output\)\s+return async_output"
        )
        # V2 keeps host timing only.
        self.assertNotIn("pd_device_timing", V2.read_text(encoding="utf-8"))

    def test_v1_t56_stage_points(self):
        """The four T56 stages: one block each, around the calls the segment is defined as."""
        v1 = runner_methods(V1)
        blocks = {name: stage_blocks(v1[name]) for name in ("execute_model", "sample_tokens")}
        self.assertEqual(sorted(blocks["execute_model"]), ["lm_head", "prepare_inputs"])
        self.assertEqual(sorted(blocks["sample_tokens"]), ["bookkeeping", "draft"])
        for name, found in blocks.items():
            for stage, nodes in found.items():
                self.assertEqual(len(nodes), 1, f"{stage} must be measured in one place in {name}")

        prepare = calls_in(blocks["execute_model"]["prepare_inputs"][0])
        for call in ("self._update_states", "self._prepare_inputs", "self._determine_batch_execution_and_padding",
                     "self._build_attention_metadata", "self._preprocess", "self.synchronize_input_prep"):
            self.assertIn(call, prepare)
        self.assertNotIn("self._model_forward", prepare)  # the forward stage keeps its own events

        lm_head = calls_in(blocks["execute_model"]["lm_head"][0])
        self.assertIn("self.model.compute_logits", lm_head)
        self.assertNotIn("self._model_forward", lm_head)

        bookkeeping = calls_in(blocks["sample_tokens"]["bookkeeping"][0])
        self.assertIn("self._bookkeeping_sync", bookkeeping)
        self.assertNotIn("self.propose_draft_token_ids", bookkeeping)

        # The draft block sits inside the closure, so it covers each of its call sites exactly once.
        draft_block = blocks["sample_tokens"]["draft"][0]
        draft = calls_in(draft_block)
        for call in ("self.propose_draft_token_ids", "self._skip_drafting", "self._copy_draft_token_ids_to_cpu"):
            self.assertIn(call, draft)
        closures = [node for node in ast.walk(v1["sample_tokens"])
                    if isinstance(node, ast.FunctionDef) and node.name == "propose_draft_token_ids"]
        self.assertEqual(len(closures), 1)
        self.assertIn(draft_block, list(ast.iter_child_nodes(closures[0])))
        self.assertGreaterEqual(len(call_spans(v1["sample_tokens"], "propose_draft_token_ids")), 2)

        # The graph mode of the step, once, next to where the runner works it out.
        self.assertEqual(len(call_spans(v1["execute_model"], "pd_device_timing.note_cudagraph_mode")), 1)

    def test_v1_t56_stages_are_not_on_the_dummy_paths(self):
        v1 = runner_methods(V1)
        for name in ("_dummy_run", "_dummy_sampler_run", "profile_run", "capture_model"):
            self.assertEqual(stage_blocks(v1[name]), {}, f"{name} must not measure a T56 stage")
            self.assertEqual(call_spans(v1[name], "pd_device_timing.note_cudagraph_mode"), [])

    def test_imports_name_the_vllm_patch_symbols(self):
        for path in (V1, V2):
            source = path.read_text(encoding="utf-8")
            self.assertIn("from vllm.v1.metrics import pd_diagnostics as diag", source)
            self.assertIn("from vllm.v1.worker.worker_base import pd_execution, pd_sample", source)


FORWARD_NAMESPACE = {
    "diag": diag,
    "pd_device_timing": pdt,
    "partial": functools.partial,
    "Any": Any,
    "IntermediateTensors": type("IntermediateTensors", (), {}),
    "torch": SimpleNamespace(Tensor=type("Tensor", (), {}), npu=SimpleNamespace(is_current_stream_capturing=lambda: False)),
    "CUDAGraphMode": SimpleNamespace(FULL="FULL"),
}


class TimedEvent:
    """NPU event stub: record() keeps the host time, so tests can see where it sat around the measured call."""

    def __init__(self):
        self.recorded_ns = None
        self.stream = None

    def record(self, stream):
        self.stream = stream
        self.recorded_ns = time.monotonic_ns()

    def query(self):
        return True

    def elapsed_time(self, other):
        return (other.recorded_ns - self.recorded_ns) / MS


class HeldRecord:
    def __init__(self, phase):
        self.phase = phase
        self.fields = {}
        self.released = False

    def set_stage_fields(self, stage, **fields):
        self.fields.setdefault(stage, {}).update(fields)
        return True

    def release(self):
        self.released = True


def device_timing(phase_box):
    events = []

    def new_event():
        events.append(TimedEvent())
        return events[-1]

    held = []

    def claim():
        held.append(HeldRecord(phase_box["phase"]))
        return held[-1]

    timing = pdt.DeviceTiming(
        enabled=True,
        new_event=new_event,
        current_stream=lambda: "compute-stream",
        capturing=lambda: False,
        claim_finish=claim,
        current_context=lambda: {"execution_id": "exec-runner"},
    )
    return timing, events, held


class RunnerDeviceEventTest(unittest.TestCase):
    """The device events of C14 sit on the compute stream right around the same calls as the host spans."""

    def test_v1_forward_events_bracket_only_the_target_model_call(self):
        namespace = dict(FORWARD_NAMESPACE)
        for enable_enpu in (False, True):
            with self.subTest(enable_enpu=enable_enpu):
                clock = Clock()
                context = SimpleNamespace(cudagraph_runtime_mode="NONE", device_metadata_executor=None, capturing=False)
                namespace["get_forward_context"] = lambda context=context: context
                forward = compile_method(V1, "_model_forward", dict(namespace))
                model = type("Model", (), {"__call__": staticmethod(clock.timed("model", 20, "hidden"))})()
                phase = {"phase": "execute"}
                timing, events, held = device_timing(phase)
                runner = SimpleNamespace(model=model, enable_enpu=enable_enpu, _pd_device_timing=timing,
                                         _update_full_graph_params_if_needed=clock.timed("update_graph_params", 15))
                timing.begin_execute()
                forward(runner, 8, input_ids="ids", positions="pos")
                start, end = events
                model_start, model_end = clock.calls["model"][0]
                update_start, update_end = clock.calls["update_graph_params"][0]
                self.assertEqual((start.stream, end.stream), ("compute-stream", "compute-stream"))
                self.assertLessEqual(start.recorded_ns, model_start)
                self.assertLessEqual(model_end, end.recorded_ns)
                self.assertTrue(update_end <= start.recorded_ns or end.recorded_ns <= update_start)
                timing.end_sample(read_now=True)  # a step without a sample phase: the next read writes it
                timing.begin_execute()
                self.assertTrue(held[0].released)
                self.assertGreaterEqual(held[0].fields["forward"]["device_duration_ns"], 20 * MS)

    def test_v1_sampling_events_bracket_only_the_sampler_call(self):
        sample = compile_method(V1, "_sample", {"diag": diag, "lmhead_tp_enable": lambda: False, "pd_device_timing": pdt})
        for algorithm, spec in (("sampler", None), ("rejection", SimpleNamespace(logits_indices=[0, 1]))):
            with self.subTest(algorithm=algorithm):
                clock = Clock()
                phase = {"phase": "execute"}
                timing, events, held = device_timing(phase)
                runner = SimpleNamespace(
                    _pd_device_timing=timing,
                    input_batch=SimpleNamespace(update_async_output_token_ids=clock.timed("update_tokens", 10),
                                                sampling_metadata="metadata", num_reqs=1),
                    sampler=clock.timed("sampler", 12, "sampled"),
                    rejection_sampler=clock.timed("rejection", 14, "accepted"),
                    _get_spec_decode_draft_probs=clock.timed("draft_probs", 16),
                )
                timing.begin_execute()
                phase["phase"] = "sample"
                timing.begin_sample()
                sample(runner, "logits", spec)
                start, end = events
                call = clock.calls["sampler" if spec is None else "rejection"][0]
                self.assertLessEqual(start.recorded_ns, call[0])
                self.assertLessEqual(call[1], end.recorded_ns)
                for other in ("update_tokens", "draft_probs") if spec is not None else ("update_tokens",):
                    window = clock.calls[other][0]
                    self.assertTrue(window[1] <= start.recorded_ns or end.recorded_ns <= window[0], other)
                timing.end_sample(read_now=True)
                sampled = [h for h in held if h.phase == "sample"][0]
                self.assertGreaterEqual(sampled.fields["sampling"]["device_duration_ns"], (12 if spec is None else 14) * MS)


if __name__ == "__main__":
    unittest.main()
