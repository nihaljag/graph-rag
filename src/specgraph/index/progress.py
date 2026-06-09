"""Console progress for GraphRAG indexing.

- ``ConsoleWorkflowCallbacks`` prints each workflow's start/end and throttled
  progress ticks to the console (the GraphRAG API otherwise uses a no-op
  reporter, so long runs look frozen).
- ``HeartbeatMonitor`` is a background thread that periodically prints a liveness
  line — current workflow, elapsed time, and LLM call counts — so even a single
  long task (e.g. a big community report) visibly makes progress.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from specgraph.logging import get_logger

log = get_logger()


class ConsoleWorkflowCallbacks:
    """Implements GraphRAG's WorkflowCallbacks Protocol with console output."""

    def __init__(self, progress_min_interval: float = 3.0) -> None:
        self.current: str | None = None
        self._wf_start = 0.0
        self._last_progress = 0.0
        self._min_interval = progress_min_interval

    def pipeline_start(self, names: list[str]) -> None:
        log.info("Pipeline: %d workflows → %s", len(names), ", ".join(names))

    def pipeline_end(self, results: list[Any]) -> None:
        log.info("Pipeline complete (%d workflow results).", len(results))

    def workflow_start(self, name: str, instance: object) -> None:
        self.current = name
        self._wf_start = time.perf_counter()
        self._last_progress = 0.0
        log.info("▶ workflow: %s", name)

    def workflow_end(self, name: str, instance: object) -> None:
        dt = time.perf_counter() - self._wf_start if self._wf_start else 0.0
        log.info("✔ workflow: %s (%.1fs)", name, dt)
        self.current = None

    def progress(self, progress: Any) -> None:
        now = time.perf_counter()
        if now - self._last_progress < self._min_interval:
            return
        self._last_progress = now
        desc = getattr(progress, "description", None) or (self.current or "progress")
        total = getattr(progress, "total_items", None)
        done = getattr(progress, "completed_items", None)
        if total:
            log.info("   … %s %s/%s", desc, done, total)
        elif done:
            log.info("   … %s %s", desc, done)

    def pipeline_error(self, error: BaseException) -> None:
        log.error("Pipeline error: %s", error)


class HeartbeatMonitor:
    """Background liveness printer; reads global LLM call stats."""

    def __init__(self, callbacks: ConsoleWorkflowCallbacks, interval: float = 15.0) -> None:
        self._callbacks = callbacks
        self._interval = max(2.0, interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def __enter__(self) -> "HeartbeatMonitor":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="specgraph-heartbeat",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        from specgraph.llm.call_log import call_stats

        while not self._stop.wait(self._interval):
            stats = call_stats()
            elapsed = time.perf_counter() - self._t0
            wf = self._callbacks.current or "…"
            log.info("⏳ working [%s] | elapsed %dm%02ds | LLM: %d done, %d in-flight, %d failed",
                     wf, int(elapsed) // 60, int(elapsed) % 60,
                     stats["done"], stats["in_flight"], stats["failed"])
