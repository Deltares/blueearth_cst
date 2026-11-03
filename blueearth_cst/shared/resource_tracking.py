"""Local, best-effort WF4 resource samples; never part of scientific identity."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable

import psutil

DIRECTORY_ENV = "CST_RESOURCE_DIRECTORY"
_MONITORS: dict[Path, ResourceMonitor] = {}


class ResourceMonitor:
    """Sample the launcher and its descendants without inspecting command lines.

    CPU and I/O totals retain the last observation of exited processes. They
    are lower bounds: children that start and exit between samples are missed.
    The initial census baselines existing processes; later descendants start
    at zero. An incomplete initial census baselines all first observations.
    RSS sums can count shared pages more than once. Samples are flushed so a
    hard-killed launcher leaves useful evidence without a fabricated finish.
    """

    def __init__(self, directory: Path, interval: float = 2.0) -> None:
        self.directory = directory
        self.interval = interval
        self.root = psutil.Process()
        self.started = time.monotonic()
        self.phase = "preparation"
        self.phases: list[dict[str, Any]] = []
        self.errors: dict[str, int] = {}
        self.processes: dict[tuple[int, float], dict[str, Any]] = {}
        self.initial_processes: set[tuple[int, float]] = set()
        self.initial_census_complete = True
        self.samples = 0
        self.collection_wall_s = 0.0
        self.peak_rss = 0
        self.min_available: int | None = None
        self.last_system_cpu: dict[str, float] | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        directory.mkdir(parents=True, exist_ok=True)
        self.handle = (directory / "resources.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        )
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _read(self, label: str, call: Callable[[], Any]) -> Any:
        """Represent unsupported or inaccessible measurements explicitly."""
        try:
            return call()
        except (psutil.Error, OSError, NotImplementedError, AttributeError):
            self.errors[label] = self.errors.get(label, 0) + 1
            return None

    def _emit(self, kind: str, **values: Any) -> None:
        """Serialize one complete timestamped observation."""
        self.handle.write(
            json.dumps(
                {
                    "kind": kind,
                    "utc_epoch_s": time.time(),
                    "elapsed_s": time.monotonic() - self.started,
                    **values,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.handle.flush()

    def start(self) -> None:
        """Record host context once and begin periodic sampling."""
        power_plan = None
        if os.name == "nt":
            try:
                result = subprocess.run(
                    ["powercfg", "/getactivescheme"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if result.returncode == 0:
                    power_plan = result.stdout.strip()
                else:
                    self.errors["power_plan"] = 1
            except (OSError, subprocess.TimeoutExpired):
                self.errors["power_plan"] = 1
        memory = self._read("virtual_memory", psutil.virtual_memory)
        self._emit(
            "host",
            schema_version="wf4-resources/1",
            interval_s=self.interval,
            platform=platform.platform(),
            cpu_model=platform.processor(),
            logical_cpus=psutil.cpu_count(),
            physical_cpus=psutil.cpu_count(logical=False),
            total_memory_bytes=None if memory is None else memory.total,
            power_plan_at_start=power_plan,
            output_volume_at_start=self._read(
                "output_volume",
                lambda: psutil.disk_usage(str(self.directory))._asdict(),
            ),
            limitations=[
                "Sampled process CPU/I/O totals are lower bounds; short-lived children may be missed.",
                "Tree RSS sums may double-count shared pages; peaks are sampled.",
                "Process I/O includes cached/logical I/O, not necessarily physical disk traffic.",
                "Reported CPU MHz may be nominal (especially on Windows), not effective frequency.",
                "No portable thermal/throttling measurement; absence of a signal is not evidence of absence.",
                "System CPU, swap and disk counters include unrelated workloads.",
            ],
        )
        self.mark("preparation")
        self.sample()
        self.thread.start()

    def mark(self, name: str) -> None:
        """Mark a launcher phase on the same clock as the resource timeline."""
        with self.lock:
            self.phase = name
            event = {"name": name, "elapsed_s": time.monotonic() - self.started}
            self.phases.append(event)
            self._emit("phase", **event)

    def sample(self) -> None:
        """Collect host pressure and individually identifiable process counters."""
        with self.lock:
            collection_started = time.monotonic()
            children = self._read(
                "children", lambda: self.root.children(recursive=True)
            )
            if self.samples == 0 and children is None:
                self.initial_census_complete = False
            rows = []
            for proc in [self.root, *(children or [])]:
                try:
                    identity = (proc.pid, proc.create_time())
                    if self.samples == 0:
                        self.initial_processes.add(identity)
                    cpu = proc.cpu_times()
                    memory = proc.memory_info()
                except psutil.Error:
                    if self.samples == 0:
                        self.initial_census_complete = False
                    self.errors["process_disappeared_or_denied"] = (
                        self.errors.get("process_disappeared_or_denied", 0) + 1
                    )
                    continue
                io = self._read("process_io", lambda: proc.io_counters())
                counters = {
                    "cpu_s": cpu.user + cpu.system,
                    "read_bytes": None if io is None else io.read_bytes,
                    "write_bytes": None if io is None else io.write_bytes,
                    "page_faults": getattr(memory, "num_page_faults", None),
                }
                if identity not in self.processes:
                    # Creation timestamps can predate our wall-clock start due
                    # to OS clock precision. Use census membership, retaining
                    # create_time only as part of the PID-reuse-safe identity.
                    preexisting = (
                        identity in self.initial_processes
                        or not self.initial_census_complete
                    )
                    baseline = {
                        key: (value if preexisting else 0)
                        for key, value in counters.items()
                    }
                    self.processes[identity] = {"baseline": baseline}
                    self._emit(
                        "process",
                        pid=proc.pid,
                        created_epoch_s=identity[1],
                        name=self._read("process_name", proc.name),
                        affinity=self._read("affinity", lambda: proc.cpu_affinity()),
                        priority=self._read("priority", proc.nice),
                    )
                self.processes[identity]["last"] = counters
                rows.append(
                    {
                        "pid": proc.pid,
                        "created_epoch_s": identity[1],
                        "rss_bytes": memory.rss,
                        "virtual_memory_bytes": memory.vms,
                        "private_memory_bytes": getattr(memory, "private", None),
                        **counters,
                    }
                )
            memory = self._read("virtual_memory", psutil.virtual_memory)
            swap = self._read("swap_memory", psutil.swap_memory)
            frequency = self._read("cpu_freq", lambda: psutil.cpu_freq())
            disk = self._read("disk_io", psutil.disk_io_counters)
            battery = self._read("battery", lambda: psutil.sensors_battery())
            system_cpu = psutil.cpu_times()._asdict()
            cpu_percent = None
            if self.last_system_cpu is not None:
                delta = {
                    key: max(0, value - self.last_system_cpu[key])
                    for key, value in system_cpu.items()
                    if key not in {"guest", "guest_nice"}
                }
                total = sum(delta.values())
                idle = delta.get("idle", 0) + delta.get("iowait", 0)
                cpu_percent = 100 * (total - idle) / total if total else None
            self.last_system_cpu = system_cpu
            self.peak_rss = max(self.peak_rss, sum(row["rss_bytes"] for row in rows))
            if memory is not None:
                self.min_available = min(
                    self.min_available
                    if self.min_available is not None
                    else memory.available,
                    memory.available,
                )
            self._emit(
                "sample",
                phase=self.phase,
                processes=rows,
                system_cpu_percent=cpu_percent,
                available_memory_bytes=None if memory is None else memory.available,
                swap=None if swap is None else swap._asdict(),
                cpu_frequency_mhz=None if frequency is None else frequency._asdict(),
                system_disk_io=None if disk is None else disk._asdict(),
                battery=None if battery is None else battery._asdict(),
                collection_wall_s=time.monotonic() - collection_started,
            )
            self.collection_wall_s += time.monotonic() - collection_started
            self.samples += 1

    def _run(self) -> None:
        """Stop collection visibly on a monitor failure without stopping WF4."""
        try:
            while not self.stop_event.wait(self.interval):
                self.sample()
        except Exception as exc:  # noqa: BLE001 -- diagnostics must not stop a run
            self.errors[f"monitor:{type(exc).__name__}"] = 1
            warnings.warn(f"WF4 resource tracking stopped: {exc}", stacklevel=2)

    def finish(self) -> dict[str, Any]:
        """Close the trace and return an explicitly sampled resource summary."""
        self.stop_event.set()
        self.thread.join()
        self.sample()
        elapsed = time.monotonic() - self.started
        cpu = sum(
            item["last"]["cpu_s"] - item["baseline"]["cpu_s"]
            for item in self.processes.values()
        )
        summary = {
            "directory": str(self.directory),
            "samples": self.samples,
            "wall_s": elapsed,
            "sampler_collection_wall_s": self.collection_wall_s,
            "sampled_tree_cpu_s": cpu,
            "sampled_mean_cpu_cores": cpu / elapsed if elapsed else None,
            "sampled_peak_tree_rss_bytes": self.peak_rss,
            "minimum_available_memory_bytes": self.min_available,
            "phases": [
                {**phase, "wall_s": end - phase["elapsed_s"]}
                for phase, end in zip(
                    self.phases,
                    [item["elapsed_s"] for item in self.phases[1:]] + [elapsed],
                    strict=True,
                )
            ],
            "measurement_errors": self.errors,
        }
        self._emit("summary", **summary)
        self.handle.close()
        return summary


def start(path: Path) -> dict[str, Any]:
    """Start WF4 tracking beside its invocation record, degrading visibly."""
    monitor = None
    try:
        monitor = ResourceMonitor(path.with_suffix("") / "diagnostics")
        monitor.start()
        _MONITORS[path] = monitor
        return {"directory": str(monitor.directory), "status": "running"}
    except Exception as exc:  # noqa: BLE001 -- diagnostics must not stop a run
        if monitor is not None:
            monitor.handle.close()
        warnings.warn(f"WF4 resource tracking unavailable: {exc}", stacklevel=2)
        return {"status": "unavailable", "error": str(exc)}


def phase(path: Path, name: str) -> None:
    """Mark a phase if tracking is active."""
    monitor = _MONITORS.get(path)
    if monitor is not None:
        try:
            monitor.mark(name)
        except OSError as exc:
            warnings.warn(f"WF4 phase tracking unavailable: {exc}", stacklevel=2)


def finish(path: Path) -> dict[str, Any] | None:
    """Finish active tracking without changing the workflow outcome."""
    monitor = _MONITORS.pop(path, None)
    if monitor is None:
        return None
    try:
        return {"status": "finished", **monitor.finish()}
    except Exception as exc:  # noqa: BLE001 -- preserve the workflow's outcome
        monitor.handle.close()
        warnings.warn(f"WF4 resource summary unavailable: {exc}", stacklevel=2)
        return {
            "status": "incomplete",
            "error": str(exc),
            "directory": str(monitor.directory),
        }


def write_batch_decision(
    batch_id: str, decision: dict[str, Any], *, threads: int, run_ids: list[str]
) -> None:
    """Save the scheduling inputs used by a batch, outside immutable outputs."""
    directory = os.environ.get(DIRECTORY_ENV)
    if not directory:
        return
    path = Path(directory) / f"batch-{batch_id}-{uuid.uuid4().hex}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "wf4-batch-resources/1",
                    **decision,
                    "rule_requested_threads": threads,
                    "run_ids": run_ids,
                    "recorded_at_epoch_s": time.time(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as exc:
        warnings.warn(f"WF4 batch resource decision unavailable: {exc}", stacklevel=2)
