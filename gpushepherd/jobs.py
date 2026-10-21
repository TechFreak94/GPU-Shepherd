"""Runs one change at a time (start/stop/restart...) in the background, with a visible log."""
from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


class Busy(RuntimeError):
    pass


@dataclass
class Job:
    id: int
    title: str
    status: str = "running"            # running | done | failed
    lines: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: float | None = None
    error: str = ""

    def log(self, line: str) -> None:
        self.lines.append(f"{time.strftime('%H:%M:%S')}  {line}")

    @property
    def seconds(self) -> int:
        return int((self.finished or time.time()) - self.started)


class JobRunner:
    def __init__(self, history: int = 15):
        self._lock = threading.Lock()
        self._next = 1
        self.current: Job | None = None
        self.history: deque[Job] = deque(maxlen=history)

    @property
    def busy(self) -> bool:
        return self.current is not None

    def submit(self, title: str, fn: Callable[[Callable[[str], None]], None], wait: bool = False) -> Job:
        with self._lock:
            if self.current is not None:
                raise Busy(f"'{self.current.title}' is still running - try again when it finishes")
            job = Job(id=self._next, title=title)
            self._next += 1
            self.current = job

        def run():
            try:
                fn(job.log)
                job.status = "done"
            except Exception as err:  # noqa: BLE001 - surfaced in the UI
                job.status = "failed"
                job.error = str(err) or err.__class__.__name__
                job.log(f"FAILED: {job.error}")
                if not str(err):
                    job.log(traceback.format_exc(limit=3))
            finally:
                job.finished = time.time()
                with self._lock:
                    self.history.appendleft(job)
                    self.current = None

        thread = threading.Thread(target=run, daemon=True, name=f"job-{job.id}")
        thread.start()
        if wait:
            thread.join()
        return job

    def recent(self) -> list[Job]:
        jobs = list(self.history)
        if self.current:
            jobs.insert(0, self.current)
        return jobs
