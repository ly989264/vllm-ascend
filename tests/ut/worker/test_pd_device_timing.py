"""Stdlib-only tests for vllm_ascend/worker/pd_device_timing.py (device-event timing of forward and sampling).

The NPU events are stubs on a scripted device timeline: the test advances device time inside the measured calls and
decides which events have completed, so every device value is exact and nothing depends on machine load. The record
claims follow vLLM's pd_claim_finish() interface: the record stays open until the timing releases it.
"""

from __future__ import annotations

import importlib.util
import random
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("pd_device_timing_ut", ROOT / "vllm_ascend/worker/pd_device_timing.py")
pdt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdt)
MS = 1_000_000
STREAM = object()


class Device:
    """A scripted device timeline; events complete only when the test says so."""

    def __init__(self):
        self.now_ms = 0.0
        self.completed = -1  # events with index <= completed have completed
        self.count = 0
        self.lock = threading.Lock()
        self.read_error = False
        self.record_error = False
        self.held: set[int] = set()

    def advance(self, ms):
        self.now_ms += ms

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
        assert stream is STREAM, "both events must be recorded on the compute stream"
        if self.device.record_error:
            raise RuntimeError("record sentinel")
        with self.device.lock:
            self.device.count += 1
            self.index = self.device.count
        self.ts = self.device.now_ms

    def query(self):
        if self.device.read_error:
            raise RuntimeError("query sentinel")
        return self.index is not None and self.index <= self.device.completed and self.index not in self.device.held

    def elapsed_time(self, other):
        assert self.query() and other.query(), "elapsed_time must not be called before both events completed"
        return other.ts - self.ts


class Claims:
    """vLLM's claim side: one claim per (execution, phase); logs fields and the write order."""

    def __init__(self):
        self.context = {"execution_id": None, "phase": None}
        self.available = True
        self.written: list[tuple] = []
        self.lock = threading.Lock()

    def claim_finish(self):
        if not self.available or self.context["execution_id"] is None:
            return None
        return Claim(self, self.context["execution_id"], self.context["phase"])


class Claim:
    def __init__(self, claims, execution_id, phase):
        self.claims, self.execution_id, self.phase = claims, execution_id, phase
        self.fields: dict = {}
        self.releases = 0
        self.written_on_error = False

    def fail(self, error_type):
        """What vLLM's wrapper does when the method raises: the record is written at once, with the error."""
        self.written_on_error = True
        with self.claims.lock:
            self.claims.written.append((self.execution_id, self.phase, {"status": "error", "error_type": error_type}))

    def set_stage_fields(self, stage, **fields):
        if self.written_on_error:
            return False
        self.fields.setdefault(stage, {}).update(fields)
        return True

    def release(self):
        if self.written_on_error:
            return
        self.releases += 1
        if self.releases == 1:
            with self.claims.lock:
                self.claims.written.append((self.execution_id, self.phase, dict(self.fields)))


def make(device=None, claims=None, capturing=False):
    device = device or Device()
    claims = claims or Claims()
    flags = {"capturing": capturing}
    timing = pdt.DeviceTiming(
        enabled=True,
        new_event=device.event,
        current_stream=lambda: STREAM,
        capturing=lambda: flags["capturing"],
        claim_finish=claims.claim_finish,
        current_context=lambda: dict(claims.context),
    )
    return timing, device, claims, flags


def step(timing, device, claims, execution_id, forward_ms, sampling_ms, synchronous):
    claims.context.update(execution_id=execution_id, phase="execute")
    timing.begin_execute()
    with timing.measure("forward"):
        device.advance(forward_ms)
    claims.context["phase"] = "sample"
    timing.begin_sample()
    with timing.measure("sampling"):
        device.advance(sampling_ms)
    if synchronous:
        device.complete_all()  # the host copied the sampled tokens back
    timing.end_sample(read_now=synchronous)


def device_values(written):
    out = {}
    for execution_id, phase, fields in written:
        for stage, values in fields.items():
            if isinstance(values, dict):
                out[(execution_id, stage)] = values.get("device_duration_ns")
    return out


class DeviceTimingTest(unittest.TestCase):
    def test_synchronous_step_writes_both_records_with_device_values(self):
        timing, device, claims, _ = make()
        step(timing, device, claims, "e1", 12.5, 3.25, synchronous=True)
        self.assertEqual([(e, p) for e, p, _ in claims.written], [("e1", "execute"), ("e1", "sample")])
        self.assertEqual(claims.written[0][2], {"forward": {"device_duration_ns": 12_500_000}})
        self.assertEqual(claims.written[1][2], {"sampling": {"device_duration_ns": 3_250_000}})

    def test_late_completion_is_read_at_the_async_output_or_at_the_next_step(self):
        for read_at in ("get_output", "next_step"):
            with self.subTest(read_at=read_at):
                timing, device, claims, _ = make()
                step(timing, device, claims, "e1", 10, 2, synchronous=False)
                timing.collect()  # get_output before the device finished
                self.assertEqual(claims.written, [])
                device.complete_all()
                if read_at == "get_output":
                    timing.collect()
                else:
                    claims.context.update(execution_id="e2", phase="execute")
                    timing.begin_execute()
                values = device_values(claims.written)
                self.assertEqual(values[("e1", "forward")], 10 * MS)
                self.assertEqual(values[("e1", "sampling")], 2 * MS)

    def test_events_that_never_complete_are_written_without_device_values(self):
        timing, device, claims, _ = make()
        step(timing, device, claims, "e1", 10, 2, synchronous=False)
        never = set(range(1, device.count + 1))  # e1's four events never complete
        device.held = never
        for n in range(2, 1 + pdt.SLOTS):
            device.complete_all()
            step(timing, device, claims, f"e{n}", n, 1, synchronous=False)
        timing.collect()
        self.assertEqual(claims.written, [])  # e1 still blocks; nothing is written out of order
        device.complete_all()
        step(timing, device, claims, f"e{1 + pdt.SLOTS}", 1 + pdt.SLOTS, 1, synchronous=False)
        # Starting the step SLOTS after e1 wrote e1 without device values, then the completed steps after it.
        written = [(e, p) for e, p, _ in claims.written]
        self.assertEqual(written[:2], [("e1", "execute"), ("e1", "sample")])
        self.assertEqual(claims.written[0][2], {})
        self.assertEqual(claims.written[1][2], {})
        device.complete_all()
        timing.collect()
        values = device_values(claims.written)
        for n in range(2, 2 + pdt.SLOTS):  # every later step, including the one that reuses e1's slot
            self.assertEqual(values[(f"e{n}", "forward")], n * MS)

    def test_a_read_error_writes_the_step_without_device_values_and_later_steps_still_get_them(self):
        timing, device, claims, _ = make()
        step(timing, device, claims, "e1", 10, 2, synchronous=False)
        device.complete_all()
        device.read_error = True
        timing.collect()  # must not raise
        self.assertEqual([f for _, _, f in claims.written], [{}, {}])
        device.read_error = False
        step(timing, device, claims, "e2", 7, 1, synchronous=True)
        self.assertEqual(device_values(claims.written)[("e2", "forward")], 7 * MS)

    def test_a_record_error_never_reaches_the_measured_call(self):
        timing, device, claims, _ = make()
        device.record_error = True
        claims.context.update(execution_id="e1", phase="execute")
        timing.begin_execute()
        ran = []
        with timing.measure("forward"):
            ran.append("model")
        self.assertEqual(ran, ["model"])
        device.record_error = False
        claims.context["phase"] = "sample"
        timing.begin_sample()
        timing.end_sample(read_now=True)
        self.assertEqual([f for _, _, f in claims.written], [{}, {}])

    def test_an_error_in_the_measured_call_propagates_unchanged(self):
        timing, device, claims, _ = make()
        claims.context.update(execution_id="e1", phase="execute")
        timing.begin_execute()
        failure = ValueError("model sentinel")
        with self.assertRaises(ValueError) as raised:
            with timing.measure("forward"):
                raise failure
        self.assertIs(raised.exception, failure)

    def test_overlapping_async_steps_get_their_own_values(self):
        timing, device, claims, _ = make()
        step(timing, device, claims, "e1", 10, 1, synchronous=False)
        step(timing, device, claims, "e2", 20, 2, synchronous=False)  # launched before e1 was read
        step(timing, device, claims, "e3", 30, 3, synchronous=False)
        device.complete_all()
        timing.collect()
        values = device_values(claims.written)
        for n in (1, 2, 3):
            self.assertEqual(values[(f"e{n}", "forward")], n * 10 * MS)
            self.assertEqual(values[(f"e{n}", "sampling")], n * MS)

    def test_records_are_written_in_step_order(self):
        timing, device, claims, _ = make()
        step(timing, device, claims, "e1", 10, 1, synchronous=False)
        first = set(range(1, device.count + 1))
        step(timing, device, claims, "e2", 20, 2, synchronous=False)
        device.complete_all()
        device.held = first  # e2 completed, e1 not yet
        timing.collect()
        self.assertEqual(claims.written, [])
        device.held = set()
        timing.collect()
        self.assertEqual([(e, p) for e, p, _ in claims.written],
                         [("e1", "execute"), ("e1", "sample"), ("e2", "execute"), ("e2", "sample")])

    def test_steps_without_events_keep_their_place(self):
        timing, device, claims, _ = make()
        step(timing, device, claims, "e1", 10, 1, synchronous=False)
        claims.context.update(execution_id="e2", phase="execute")  # a step that returns without a forward
        timing.begin_execute()
        claims.context.update(execution_id="e3", phase="execute")
        timing.begin_execute()
        self.assertEqual(claims.written, [])  # e1 not read yet, so e2 waits too
        device.complete_all()
        timing.collect()
        self.assertEqual([e for e, _, _ in claims.written], ["e1", "e1", "e2"])

    def test_a_step_whose_record_was_written_on_an_error_is_not_written_again(self):
        timing, device, claims, _ = make()
        step(timing, device, claims, "e1", 10, 1, synchronous=False)
        claims.context.update(execution_id="e2", phase="execute")
        timing.begin_execute()
        with self.assertRaises(RuntimeError):
            with timing.measure("forward"):
                device.advance(5)
                raise RuntimeError("execute sentinel")
        timing._steps[-1].claims["execute"].fail("RuntimeError")  # vLLM wrote e2's record when execute_model raised
        step(timing, device, claims, "e3", 30, 3, synchronous=False)
        device.complete_all()
        timing.collect()
        e2 = [entry for entry in claims.written if entry[0] == "e2"]
        self.assertEqual(e2, [("e2", "execute", {"status": "error", "error_type": "RuntimeError"})])
        values = device_values(claims.written)
        self.assertEqual(values[("e1", "forward")], 10 * MS)
        self.assertEqual(values[("e3", "forward")], 30 * MS)
        self.assertEqual([e for e, _, _ in claims.written if e != "e2"], ["e1", "e1", "e3", "e3"])

    def test_a_stage_measured_twice_in_one_step_gets_no_device_value(self):
        timing, device, claims, _ = make()
        claims.context.update(execution_id="e1", phase="execute")
        timing.begin_execute()
        for _ in range(2):
            with timing.measure("forward"):
                device.advance(5)
        claims.context["phase"] = "sample"
        timing.begin_sample()
        device.complete_all()
        timing.end_sample(read_now=True)
        self.assertEqual(claims.written[0][2], {})

    def test_nothing_is_held_or_recorded_off_rank_0_during_capture_or_without_a_record(self):
        self.assertFalse(pdt.DeviceTiming.for_runner(1).enabled)
        timing, device, claims, flags = make(capturing=True)
        step(timing, device, claims, "e1", 10, 1, synchronous=True)
        self.assertEqual((device.count, claims.written), (0, []))
        timing, device, claims, _ = make()
        claims.available = False
        step(timing, device, claims, "e1", 10, 1, synchronous=True)
        self.assertEqual((device.count, claims.written), (0, []))

    def test_release_all_writes_held_records_without_device_values(self):
        timing, device, claims, _ = make()
        step(timing, device, claims, "e1", 10, 1, synchronous=False)
        timing.release_all()
        self.assertEqual([f for _, _, f in claims.written], [{}, {}])

    def test_get_output_reads_after_the_copy_and_never_changes_or_breaks_it(self):
        class Base:
            def __init__(self, value):
                self.value = value

            def get_output(self):
                return self.value

        cls = pdt.async_output_class(Base)
        timing, device, claims, _ = make()
        step(timing, device, claims, "e1", 10, 1, synchronous=False)
        output = cls("sentinel")
        output.pd_device_timing = timing
        device.complete_all()
        self.assertEqual(output.get_output(), "sentinel")
        self.assertEqual(device_values(claims.written)[("e1", "forward")], 10 * MS)

        class Broken:
            def collect(self):
                raise RuntimeError("diagnostics sentinel")

        output = cls("still the output")
        output.pd_device_timing = Broken()
        self.assertEqual(output.get_output(), "still the output")
        failure = RuntimeError("copy sentinel")

        class Failing(Base):
            def get_output(self):
                raise failure

        output = pdt.async_output_class(Failing)("x")
        output.pd_device_timing = timing
        with self.assertRaises(RuntimeError) as raised:
            output.get_output()
        self.assertIs(raised.exception, failure)

    def test_concurrent_reader_and_writer(self):
        timing, device, claims, _ = make()
        self.assertIsInstance(timing._lock, type(threading.Lock()))
        stop = threading.Event()
        errors = []

        def reader():
            while not stop.is_set():
                try:
                    timing.collect()
                except Exception as exc:  # pragma: no cover - would fail the test
                    errors.append(exc)

        thread = threading.Thread(target=reader)
        thread.start()
        rng = random.Random(7)
        try:
            for n in range(1, 301):
                step(timing, device, claims, f"e{n}", n, n / 10, synchronous=False)
                if rng.random() < 0.7:
                    device.complete_all()
        finally:
            device.complete_all()
            timing.collect()
            stop.set()
            thread.join()
        self.assertEqual(errors, [])
        order = [(e, p) for e, p, _ in claims.written]
        expected = [(f"e{n}", p) for n in range(1, 301) for p in ("execute", "sample")]
        self.assertEqual(order, expected)  # every record once, in step order
        values = device_values(claims.written)
        for n in range(1, 301):
            if values.get((f"e{n}", "forward")) is not None:  # read in time: exactly its own value
                self.assertEqual(values[(f"e{n}", "forward")], n * MS)


class RunnerHookTest(unittest.TestCase):
    def test_hooks_never_raise(self):
        class Runner:
            pass

        runner = Runner()
        pdt.begin_execute(runner)  # no vLLM here: stays disabled
        with pdt.measure(runner, "forward"):
            pass
        pdt.begin_sample(runner)
        pdt.end_sample(runner, read_now=True)

        class Exploding:
            enabled = True

            def __getattr__(self, name):
                raise RuntimeError("timing sentinel")

        runner._pd_device_timing = Exploding()
        pdt.begin_execute(runner)
        pdt.begin_sample(runner)
        pdt.end_sample(runner, read_now=False, async_output=object())
        with pdt.measure(runner, "forward"):
            pass


if __name__ == "__main__":
    unittest.main()
