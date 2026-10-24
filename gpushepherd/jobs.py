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


class Cancelled(RuntimeError):
    """Raised by a job that was called off before it changed anything."""


@dataclass
class Job:
    id: int
    title: str
    status: str = "running"            # running | done | failed | cancelled
    lines: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: float | None = None
    error: str = ""
    # instance id -> what it is still doing, while the job waits for it before stopping it
    waiting: dict[str, dict] = field(default_factory=dict)
    skip: threading.Event = field(default_factory=threading.Event)     # "Stop now": stop waiting
    cancel: threading.Event = field(default_factory=threading.Event)   # "Cancel": call the change off

    def log(self, line: str) -> None:
        self.lines.append(f"{time.strftime('%H:%M:%S')}  {line}")

    __call__ = log                     # a job can be passed wherever a log function is expected

    def call_off(self) -> None:
        self.cancel.set()
        self.skip.set()                # wakes the waiting loop at once

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

    def submit(self, title: str, fn: Callable[[Job], None], wait: bool = False) -> Job:
        """Run fn(job) in the background. The job is callable, so fn can treat it as a log function."""
        with self._lock:
            if self.current is not None:
                raise Busy(f"'{self.current.title}' is still running - try again when it finishes")
            job = Job(id=self._next, title=title)
            self._next += 1
            self.current = job

        def run():
            try:
                fn(job)
                job.status = "done"
            except Cancelled as err:
                job.status = "cancelled"
                job.error = str(err)
                job.log(f"Cancelled: {job.error}")
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
