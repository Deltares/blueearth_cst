"""Run a Windows child in a job that dies with its supervising parent.

The project lock is owned by the parent. A suspended child is assigned before
it can launch descendants, and closing the sole job handle kills the tree.
"""

import codecs
import ctypes
import os
import shutil
import subprocess
import sys
import threading
from ctypes import wintypes
from pathlib import Path
from typing import Mapping, Sequence


class _BasicLimit(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimit),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def run_owned_child(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str]
) -> int:
    """Return child exit code; refuse if atomic job assignment is unavailable."""
    if os.name != "nt":
        raise OSError("Windows process-tree ownership is required for this launcher")
    import _winapi

    executable = shutil.which(command[0], path=env.get("PATH"))
    if executable is None:
        raise FileNotFoundError(f"project child executable not found: {command[0]}")

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel.ResumeThread.restype = wintypes.DWORD
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    read_fd = write_fd = input_fd = None
    reader = None
    startup = subprocess.STARTUPINFO()
    capture = bool(env.get("CST_EXECUTION_DIRECTORY"))
    if capture:
        import msvcrt

        read_fd, write_fd = os.pipe()
        input_fd = os.open(os.devnull, os.O_RDONLY)
        output_handle = msvcrt.get_osfhandle(write_fd)
        input_handle = msvcrt.get_osfhandle(input_fd)
        os.set_handle_inheritable(output_handle, True)
        os.set_handle_inheritable(input_handle, True)
        startup.dwFlags |= subprocess.STARTF_USESTDHANDLES
        startup.hStdInput = input_handle
        startup.hStdOutput = startup.hStdError = output_handle
        startup.lpAttributeList = {"handle_list": [output_handle, input_handle]}
    try:
        limits = _ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(
            job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        process = thread = None
        try:
            process, thread, _, _ = _winapi.CreateProcess(
                executable,
                subprocess.list2cmdline(list(command)),
                None,
                None,
                capture,
                0x00000004 | 0x00000400,  # SUSPENDED | UNICODE_ENVIRONMENT
                dict(env),
                str(cwd),
                startup,
            )
            if not kernel.AssignProcessToJobObject(job, int(process)):
                raise ctypes.WinError(ctypes.get_last_error())
            if kernel.ResumeThread(int(thread)) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            if capture:
                os.close(write_fd)
                write_fd = None
                reader = threading.Thread(
                    target=_relay_output, args=(read_fd, sys.stdout), daemon=True
                )
                reader.start()
            _winapi.WaitForSingleObject(process, _winapi.INFINITE)
            return _winapi.GetExitCodeProcess(process)
        finally:
            if thread is not None:
                _winapi.CloseHandle(thread)
            if process is not None:
                _winapi.CloseHandle(process)
    finally:
        kernel.CloseHandle(job)
        for descriptor in (write_fd, input_fd):
            if descriptor is not None:
                os.close(descriptor)
        if reader is not None:
            reader.join()
        elif read_fd is not None:
            os.close(read_fd)


def _relay_output(descriptor, stream):
    """Stream merged child output without buffering whole jobs in memory."""
    try:
        _relay_chunks(iter(lambda: os.read(descriptor, 65536), b""), stream)
    finally:
        os.close(descriptor)


def _relay_chunks(chunks, stream):
    """Decode UTF-8 and normalize CRLF once, retaining standalone redraws.

    Text console streams translate newlines on Windows. Passing through CRLF
    unchanged would add another CR at every nested launcher and corrupt rows.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    for chunk in chunks:
        text = pending + decoder.decode(chunk)
        pending = "\r" if text.endswith("\r") else ""
        if pending:
            text = text[:-1]
        stream.write(text.replace("\r\n", "\n"))
        stream.flush()
    stream.write((pending + decoder.decode(b"", final=True)).replace("\r\n", "\n"))
    stream.flush()


def run_project_child(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], writing: bool = True
) -> int:
    """Run a project child, refusing unsupported writer lifetime semantics."""
    if command and Path(command[0]).name.lower() in {"snakemake", "snakemake.exe"}:
        command = [
            sys.executable,
            str(Path(__file__).with_name("snakemake_console.py")),
            *command[1:],
        ]
    elif list(command[1:3]) == ["-m", "snakemake"]:
        command = [
            command[0],
            str(Path(__file__).with_name("snakemake_console.py")),
            *command[3:],
        ]
    if os.name == "nt":
        return run_owned_child(command, cwd=cwd, env=env)
    if writing:
        raise OSError(
            "project writer descendants lack a verified lifetime guarantee on this platform"
        )
    if env.get("CST_EXECUTION_DIRECTORY"):
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        with process.stdout:
            _relay_chunks(iter(lambda: process.stdout.read1(65536), b""), sys.stdout)
        return process.wait()
    return subprocess.run(command, cwd=cwd, env=env, check=False).returncode
