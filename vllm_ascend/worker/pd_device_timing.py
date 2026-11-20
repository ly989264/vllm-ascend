"""Device-event timing of the V1 runner's forward and sampling, for the [PD_DIAG] worker_summary records.

On tp_rank 0 only, each step records a pair of NPU events, on the current compute stream, right around the target
model call ("forward") and around the sampler call ("sampling"). The step's two worker_summary records (phase
"execute" and "sample") are held open with vLLM's pd_claim_finish() and written, strictly in step order, once the
events are read; the value goes into stages.forward / stages.sampling as device_duration_ns, next to the unchanged
host duration_ns. Reading never waits: query() both events first, elapsed_time() only when both completed.

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
import threading
from collections import deque
from contextlib import contextmanager
from typing import Any

SLOTS = 4  # event pairs kept per stage; also the number of steps a read may lag
_PHASE_OF_STAGE = {"forward": "execute", "sampling": "sample"}


class _Step:
    __slots__ = ("seq", "execution_id", "slot", "claims", "recorded", "repeated", "settled", "broken")

    def __init__(self, seq: int, execution_id: Any, slot: int | None):
        self.seq = seq
        self.execution_id = execution_id
        self.slot = slot
        self.claims: dict[str, Any] = {}
        self.recorded: set[str] = set()
        self.repeated: set[str] = set()
        self.settled = False
        self.broken = False


class DeviceTiming:
    def __init__(self, *, enabled: bool, new_event=None, current_stream=None, capturing=None, claim_finish=None,
                 current_context=None):
        self.enabled = enabled
        self._new_event = new_event
        self._current_stream = current_stream
        self._capturing = capturing
        self._claim_finish = claim_finish
        self._current_context = current_context
        self._lock = threading.Lock()
        self._steps: deque[_Step] = deque()
        self._events: list[dict[str, tuple[Any, Any]]] = [{} for _ in range(SLOTS)]
        self._owner: list[_Step | None] = [None] * SLOTS
        self._seq = 0
        if enabled:
            atexit.register(self.release_all)

    @classmethod
    def for_runner(cls, tp_rank: int) -> DeviceTiming:
        if tp_rank != 0:
            return cls(enabled=False)
        try:
            import torch
            from vllm.v1.metrics import pd_diagnostics as diag
            from vllm.v1.worker.worker_base import pd_claim_finish

            npu = torch.npu
            return cls(
                enabled=True,
                new_event=lambda: npu.Event(enable_timing=True),
                current_stream=npu.current_stream,
                capturing=npu.is_current_stream_capturing,
                claim_finish=pd_claim_finish,
                current_context=diag.current,
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
                self._steps.append(step)
        except Exception:
            self._disable()

    @contextmanager
    def measure(self, stage: str):
        """Around the target model call ("forward") or the sampler call ("sampling")."""
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

    def _collect_locked(self) -> None:
        while self._steps:
            step = self._steps[0]
            if not step.settled:
                return
            values: dict[str, int] = {}
            done = True
            if not step.broken and step.slot is not None:
                try:
                    for stage in sorted(step.recorded):
                        start, end = self._events[step.slot][stage]
                        if not (end.query() and start.query()):
                            done = False
                            break
                        ms = start.elapsed_time(end)
                        if isinstance(ms, (int, float)) and not isinstance(ms, bool) and math.isfinite(ms) and ms >= 0:
                            values[stage] = int(round(ms * 1_000_000))
                except Exception:
                    step.broken = True
            if not done and not step.broken and self._seq - step.seq < SLOTS:
                return  # not yet; a later read point takes it
            if done and not step.broken:
                for stage, ns in values.items():
                    claim = step.claims.get(_PHASE_OF_STAGE[stage])
                    if claim is not None:
                        try:
                            claim.set_stage_fields(stage, device_duration_ns=ns)
                        except Exception:
                            pass
            self._release_locked(step)

    def _release_locked(self, step: _Step) -> None:
        self._steps.popleft()
        if step.slot is not None and self._owner[step.slot] is step:
            self._owner[step.slot] = None
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
    timing = DeviceTiming.for_runner(tp_rank)
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
