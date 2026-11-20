"""Device-event timing of the V1 runner's steps, for the [PD_DIAG] worker_summary records.

On tp_rank 0 only, each step records a pair of NPU events, on the current compute stream, right around the target
model call ("forward") and around the sampler call ("sampling"). The step's two worker_summary records (phase
"execute" and "sample") are held open with vLLM's pd_claim_finish() and written, strictly in step order, once the
events are read; the value goes into stages.<name> as device_duration_ns, next to the unchanged host duration_ns.
Reading never waits: query() both events first, elapsed_time() only when both completed.

Four optional stages cover the rest of the step -- "prepare_inputs", "lm_head" (execute), "draft",
"bookkeeping" (sample) -- through stage(), which opens the host span and the device events together. They are
written only on a PD decode node (is_pd_decode()) and only for the stages VLLM_PD_DEVICE_STAGES leaves on, so a
prefill node's records stay byte-identical to what forward and sampling alone wrote. The six stages run one after
another on the same compute stream, so their device intervals do not overlap and may be summed.

Where the events are read:
  - synchronous runs (P): when sample_tokens returns; the sampled tokens were already copied to the host there;
  - asynchronous runs (D): in the async output's get_output(), after its copy to the host completed (the copy stream
    waited for the compute stream, so the step's events completed too); that is the thread that consumes the output;
  - fallback: at the start of the next execute_model, and at process exit.
A step whose events are not done SLOTS steps later, or that hits any error, is written without device_duration_ns.
Nothing here raises into inference; on an unexpected error the timing switches itself off and writes what it holds.
"""

from __future__ import annotations

import atexit
import math
import os
import threading
from collections import deque
from contextlib import contextmanager
from typing import Any

SLOTS = 4  # event pairs kept per stage; also the number of steps a read may lag
# The order the stages run in within one step; also the order the startup line lists them.
STAGE_ORDER = ("forward", "sampling", "prepare_inputs", "lm_head", "draft", "bookkeeping")
# These two are always measured; VLLM_PD_DEVICE_STAGES never touches them.
BASE_STAGES = ("forward", "sampling")
# These are optional; each one can be switched off, and all of them are off unless this is a decode node.
OPTIONAL_STAGES = ("prepare_inputs", "lm_head", "draft", "bookkeeping")
_PHASE_OF_STAGE = {
    "forward": "execute",
    "sampling": "sample",
    "prepare_inputs": "execute",
    "lm_head": "execute",
    "draft": "sample",
    "bookkeeping": "sample",
}
_ENV_STAGES = "VLLM_PD_DEVICE_STAGES"      # unset: all four; a comma-separated subset; "none": the two above only
# The envelope of one step: the start of the first stage to the end of the last one.
_SPAN_FROM, _SPAN_TO = "prepare_inputs", "draft"
# Measured at every granularity, but its stage value goes into the record only at the full one. The device events
# are recorded either way, so the envelope above and what a step costs on the device do not move with the level.
_VALUE_AT_FULL_LEVEL_ONLY = ("bookkeeping",)
FULL_LEVEL = 3


def _parse_stages(raw: str | None) -> tuple[tuple[str, ...], str]:
    """(stages to measure, where that came from). Unknown names are ignored; the startup line shows what took
    effect. forward/sampling are not switchable, so naming them here does nothing."""
    if raw is None:
        return OPTIONAL_STAGES, "default"
    wanted = {part.strip() for part in raw.split(",")}
    if "none" in {part.lower() for part in wanted}:
        return (), "env"
    return tuple(name for name in OPTIONAL_STAGES if name in wanted), "env"


class _Step:
    __slots__ = ("seq", "execution_id", "slot", "claims", "batches", "recorded", "repeated", "settled", "broken")

    def __init__(self, seq: int, execution_id: Any, slot: int | None):
        self.seq = seq
        self.execution_id = execution_id
        self.slot = slot
        self.claims: dict[str, Any] = {}
        self.batches: dict[str, Any] = {}
        self.recorded: set[str] = set()
        self.repeated: set[str] = set()
        self.settled = False
        self.broken = False


class DeviceTiming:
    def __init__(self, *, enabled: bool, new_event=None, current_stream=None, capturing=None, claim_finish=None,
                 current_context=None, current_batch=None, stages: tuple[str, ...] = (), stages_source: str = "default",
                 decode_node: bool = False, level: int = FULL_LEVEL):
        self.enabled = enabled
        self._new_event = new_event
        self._current_stream = current_stream
        self._capturing = capturing
        self._claim_finish = claim_finish
        self._current_context = current_context
        self._current_batch = current_batch
        self.decode_node = bool(decode_node) and enabled
        self.stages = frozenset(stages) if self.decode_node else frozenset()
        self.level = level
        self._stages_source = stages_source
        self._lock = threading.Lock()
        self._steps: deque[_Step] = deque()
        self._events: list[dict[str, tuple[Any, Any]]] = [{} for _ in range(SLOTS)]
        self._owner: list[_Step | None] = [None] * SLOTS
        self._seq = 0
        if enabled:
            atexit.register(self.release_all)

    @property
    def runner_stages(self) -> tuple[str, ...]:
        """What the Worker startup line lists, in step order. Empty on anything but a decode node, so the line --
        and the records -- stay byte-identical to forward and sampling alone there. On a decode node it always
        lists at least forward and sampling, so "switched off here" (forward,sampling) reads differently from
        "not a decode node" (no runner_stages on the line at all)."""
        if not self.decode_node:
            return ()
        return tuple(name for name in STAGE_ORDER if name in BASE_STAGES or name in self.stages)

    @property
    def stages_source(self) -> str:
        return self._stages_source

    def reports(self, name: str) -> bool:
        """Whether this stage's value goes into the record. A stage this says no to is still measured: the host
        span is not opened and the device value is not written, but both device events are recorded."""
        return self.level >= FULL_LEVEL or name not in _VALUE_AT_FULL_LEVEL_ONLY

    @classmethod
    def switches(cls, diag: Any, vllm_config: Any) -> tuple[tuple[str, ...], str, bool, int]:
        """What this process measures beyond the two base stages: (stages, where it came from, is this a decode
        node, the granularity). A prefill node -- or no way to tell -- measures none of it, whatever the
        environment says. The granularity decides which measured stages reach the record, never which ones are
        measured, so the stages here and the startup line read the same at every level."""
        stages, source = _parse_stages(os.environ.get(_ENV_STAGES))
        level = _diag_level(diag)
        if not _is_decode_node(diag, vllm_config):
            return (), source, False, level
        return stages, source, True, level

    @classmethod
    def for_runner(cls, tp_rank: int, vllm_config: Any = None) -> DeviceTiming:
        if tp_rank != 0:
            return cls(enabled=False)
        try:
            import torch
            from vllm.v1.metrics import pd_diagnostics as diag
            from vllm.v1.worker.worker_base import pd_claim_finish

            stages, source, decode, level = cls.switches(diag, vllm_config)
            npu = torch.npu
            return cls(
                enabled=True,
                new_event=lambda: npu.Event(enable_timing=True),
                current_stream=npu.current_stream,
                capturing=npu.is_current_stream_capturing,
                claim_finish=pd_claim_finish,
                current_context=diag.current,
                current_batch=diag.current_batch,
                stages=stages,
                stages_source=source,
                decode_node=decode,
                level=level,
            )
        except Exception:  # no NPU events or no claim interface: host timing only
            return cls(enabled=False)

    # ---- called by the runner (main thread) ----------------------------------------------------------------------
    def begin_execute(self) -> None:
        """Start of execute_model: fallback read, then hold this step's execute record."""
        if not self.enabled:
            return
        try:
            with self._lock:
                for step in self._steps:  # a new step: no sample record joins an earlier one
                    step.settled = True
                self._seq += 1  # counts this step, so a step SLOTS older gives up its slot before it is reused
                self._collect_locked()
                if self._capturing():
                    return
                claim = self._claim_finish()
                if claim is None:
                    return
                index = self._seq % SLOTS
                slot = index if self._owner[index] is None else None
                step = _Step(self._seq, self._execution_id(), slot)
                if slot is not None:
                    self._owner[slot] = step
                step.claims["execute"] = claim
                self._hold_batch_locked(step, "execute")
                self._steps.append(step)
        except Exception:
            self._disable()

    @contextmanager
    def measure(self, stage: str):
        """Around the call this stage times. The two record() calls are what the step pays on the host for it."""
        pair = None
        if self.enabled:
            try:
                pair = self._pair(stage)
                if pair is not None:
                    pair[0].record(self._current_stream())
            except Exception:
                pair = None
                self._mark_broken()
        yield
        if pair is not None:
            try:
                pair[1].record(self._current_stream())
                with self._lock:
                    step = self._current_step_locked()
                    if step is not None:
                        step.recorded.add(stage)
            except Exception:
                self._mark_broken()

    def begin_sample(self) -> None:
        """Start of sample_tokens' main path: hold this step's sample record."""
        if not self.enabled:
            return
        try:
            with self._lock:
                step = self._current_step_locked()
                if step is None or step.settled or "sample" in step.claims:
                    return
                claim = self._claim_finish()
                if claim is not None:
                    step.claims["sample"] = claim
                    self._hold_batch_locked(step, "sample")
        except Exception:
            self._disable()

    def end_sample(self, read_now: bool) -> None:
        """sample_tokens is returning. read_now: the host already waited for this step (synchronous runs)."""
        if not self.enabled:
            return
        try:
            with self._lock:
                step = self._current_step_locked()
                if step is not None:
                    step.settled = True
                if read_now:
                    self._collect_locked()
        except Exception:
            self._disable()

    # ---- called from the thread that consumes the async output, or anywhere ---------------------------------------
    def collect(self) -> None:
        """Read completed steps and write their records in step order."""
        if not self.enabled:
            return
        try:
            with self._lock:
                self._collect_locked()
        except Exception:
            self._disable()

    def release_all(self) -> None:
        """Write every held record without device values (process exit, or after an unexpected error)."""
        try:
            with self._lock:
                while self._steps:
                    self._release_locked(self._steps[0])
        except Exception:
            pass

    # ---- internals -------------------------------------------------------------------------------------------------
    def _execution_id(self):
        try:
            return (self._current_context() or {}).get("execution_id")
        except Exception:
            return None

    def _hold_batch_locked(self, step: _Step, phase: str) -> None:
        """Keep this phase's open record, for the step's own fields (the claim only reaches the stages)."""
        if not self.stages or self._current_batch is None:
            return
        try:
            batch = self._current_batch()
        except Exception:
            return
        if batch is not None:
            step.batches[phase] = batch

    def _current_step_locked(self) -> _Step | None:
        if not self._steps:
            return None
        step = self._steps[-1]
        return step if step.execution_id == self._execution_id() else None

    def _pair(self, stage: str):
        if self._capturing():
            return None
        with self._lock:
            step = self._current_step_locked()
            if step is None or step.slot is None or step.broken or step.settled:
                return None
            if stage in step.recorded or stage in step.repeated:
                # Called twice in one step: the host value sums both calls; one pair cannot, so no device value.
                step.repeated.add(stage)
                step.recorded.discard(stage)
                return None
            events = self._events[step.slot]
            if stage not in events:
                events[stage] = (self._new_event(), self._new_event())
            return events[stage]

    def _mark_broken(self) -> None:
        try:
            with self._lock:
                step = self._current_step_locked()
                if step is not None:
                    step.broken = True
        except Exception:
            pass

    @staticmethod
    def _duration_ns(start: Any, end: Any) -> int | None:
        ms = start.elapsed_time(end)
        if isinstance(ms, (int, float)) and not isinstance(ms, bool) and math.isfinite(ms) and ms >= 0:
            return int(round(ms * 1_000_000))
        return None

    def _collect_locked(self) -> None:
        while self._steps:
            step = self._steps[0]
            if not step.settled:
                return
            values: dict[str, int] = {}
            span_ns = None
            done = True
            if not step.broken and step.slot is not None:
                try:
                    events = self._events[step.slot]
                    for stage in sorted(step.recorded):
                        start, end = events[stage]
                        if not (end.query() and start.query()):
                            done = False
                            break
                        value = self._duration_ns(start, end)
                        if value is not None:
                            values[stage] = value
                    if done and _SPAN_FROM in step.recorded and _SPAN_TO in step.recorded:
                        # The whole step on the device: the first stage's start event to the last one's end event.
                        span_ns = self._duration_ns(events[_SPAN_FROM][0], events[_SPAN_TO][1])
                except Exception:
                    step.broken = True
            if not done and not step.broken and self._seq - step.seq < SLOTS:
                return  # not yet; a later read point takes it
            if done and not step.broken:
                for stage, ns in values.items():
                    if not self.reports(stage):
                        continue
                    claim = step.claims.get(_PHASE_OF_STAGE[stage])
                    if claim is not None:
                        try:
                            claim.set_stage_fields(stage, device_duration_ns=ns)
                        except Exception:
                            pass
                if span_ns is not None:
                    self._update_batch(step, "sample", step_device_span_ns=span_ns)
            self._release_locked(step)

    @staticmethod
    def _update_batch(step: _Step, phase: str, **fields: Any) -> None:
        """A field on the step's own record. Ignored once that record has been written."""
        batch = step.batches.get(phase)
        if batch is None:
            return
        try:
            batch.update(**fields)
        except Exception:
            pass

    def _release_locked(self, step: _Step) -> None:
        self._steps.popleft()
        if step.slot is not None and self._owner[step.slot] is step:
            self._owner[step.slot] = None
        step.batches.clear()
        for phase in ("execute", "sample"):
            claim = step.claims.get(phase)
            if claim is not None:
                try:
                    claim.release()
                except Exception:
                    pass

    def _disable(self) -> None:
        self.enabled = False
        self.release_all()


# ---- runner hooks: one call per site, never raising ------------------------------------------------------------------
_DISABLED = DeviceTiming(enabled=False)


def _diag_level(diag: Any) -> int:
    """The granularity the vLLM side settled on, read once per process together with the stages. A vLLM patch that
    does not offer one leaves this at the full level, so its behaviour is the one it had."""
    try:
        read = getattr(diag, "diag_level", None)
        if read is None:
            return FULL_LEVEL
        answer = read()
        # It answers with (level, source); the source is for the line the vLLM side prints, not for here.
        value = getattr(answer, "level", answer)
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= FULL_LEVEL:
            return value
    except Exception:
        pass
    return FULL_LEVEL


def _is_decode_node(diag: Any, vllm_config: Any) -> bool:
    """vllm-ascend's own condition, reached through the shared function on the vLLM side. Also requires the four
    spans to be registered there, so an older vLLM side leaves this at the base behaviour instead of a half-on
    state (the Worker startup line then lists no runner_stages, which the upgrade check catches)."""
    try:
        is_pd_decode = getattr(diag, "is_pd_decode", None)
        if is_pd_decode is None or not is_pd_decode(vllm_config):
            return False
        spans = getattr(diag, "_SPANS", {})
        return all("runner." + name in spans for name in OPTIONAL_STAGES)
    except Exception:
        return False


def _timing(runner, create: bool = False) -> DeviceTiming:
    timing = getattr(runner, "_pd_device_timing", None)
    if timing is not None or not create:
        return timing or _DISABLED
    try:
        from vllm.v1.metrics import pd_diagnostics as diag

        tp_rank = diag.current().get("tp_rank")
    except Exception:
        return _DISABLED
    if tp_rank is None:  # outside a worker_summary step: decide later
        return _DISABLED
    timing = DeviceTiming.for_runner(tp_rank, getattr(runner, "vllm_config", None))
    try:
        runner._pd_device_timing = timing
    except Exception:
        return _DISABLED
    return timing


def begin_execute(runner) -> None:
    try:
        _timing(runner, create=True).begin_execute()
    except Exception:
        pass


def measure(runner, stage: str):
    try:
        return _timing(runner).measure(stage)
    except Exception:
        return _DISABLED.measure(stage)


@contextmanager
def stage(runner, name: str):
    """An optional stage: the host span and the device events together, or nothing at all when it is off. Off is
    the whole stage, so a node that does not measure it writes the same record as forward and sampling alone."""
    span = None
    measured = None
    try:
        timing = _timing(runner)
        if name in timing.stages:
            measured = timing.measure(name)
            if timing.reports(name):
                from vllm.v1.metrics import pd_diagnostics as diag

                span = diag.span("runner." + name)
    except Exception:
        span = None
    if span is None and measured is None:
        yield
        return
    if span is None:  # measured, but its value is not reported at this granularity
        with measured:
            yield
        return
    if measured is None:
        with span:
            yield
        return
    with span, measured:
        yield


def note_cudagraph_mode(runner, stats: Any = None, mode: Any = None) -> None:
    """The graph mode this step's target model actually ran in, onto the open execute record. Same gate as the optional
    stages, so a node that measures none of them writes the same record as forward and sampling alone."""
    try:
        if not _timing(runner).stages:
            return
        value = getattr(stats, "runtime_mode", None)
        if value is None:
            value = getattr(mode, "name", None)
        if value is None:
            return
        from vllm.v1.metrics import pd_diagnostics as diag

        batch = diag.current_batch()
        if batch is not None:
            batch.update(cudagraph_runtime_mode=str(value))
    except Exception:
        pass


def begin_sample(runner) -> None:
    try:
        _timing(runner).begin_sample()
    except Exception:
        pass


def end_sample(runner, read_now: bool, async_output=None) -> None:
    try:
        timing = _timing(runner)
        if async_output is not None and timing.enabled:
            async_output.pd_device_timing = timing
        timing.end_sample(read_now)
    except Exception:
        pass


_ASYNC_CLASSES: dict[type, type] = {}


def async_output_class(base: type) -> type:
    """base (vLLM's AsyncGPUModelRunnerOutput) with get_output() also reading this step's events afterwards."""
    cls = _ASYNC_CLASSES.get(base)
    if cls is None:

        class PdAsyncModelRunnerOutput(base):  # type: ignore[valid-type, misc]
            pd_device_timing = None

            def get_output(self):
                output = super().get_output()
                try:
                    timing = self.pd_device_timing
                    if timing is not None:
                        timing.collect()
                except Exception:
                    pass
                return output

        cls = _ASYNC_CLASSES[base] = PdAsyncModelRunnerOutput
    return cls
