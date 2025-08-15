import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import psutil

from tlopo_toolkit.exceptions import ProcessSpawnError, ProcessTerminationError


@dataclass(frozen=True)
class Executable:
    """Executable configuration."""

    path: Path
    working_directory: Optional[Path] = None

    @property
    def cwd(self) -> Path:
        """Get working directory, defaulting to executable parent."""
        return self.working_directory or self.path.parent


def spawn_process(executable: Executable, args: Optional[list[str]] = None) -> psutil.Process:
    """Spawn a new process from executable."""
    cmd: list[str] = [str(executable.path)] + (args or [])
    try:
        proc: subprocess.Popen[bytes] = subprocess.Popen(cmd, cwd=executable.cwd)
        return psutil.Process(proc.pid)
    except (OSError, subprocess.SubprocessError) as e:
        raise ProcessSpawnError(f"Failed to spawn: {e}") from e


def find_process(executable: Executable) -> Optional[psutil.Process]:
    """Find a process with the given executable path."""
    for process in psutil.process_iter():
        try:
            if Path(process.exe()) == executable.path:
                return process
        except psutil.AccessDenied as e:
            pass
    return None


def find_all_processes(executable: Executable) -> list[psutil.Process]:
    """Find all processes with the given executable path."""
    processes: list[psutil.Process] = []
    for process in psutil.process_iter():
        try:
            if Path(process.exe()) == executable.path:
                processes.append(process)
        except psutil.AccessDenied as e:
            pass
    return processes


def terminate_process(process: psutil.Process, force: bool = False, timeout: float = 5.0) -> None:
    """Terminate a process."""
    if not process.is_running():
        return

    try:
        if force:
            process.kill()
        else:
            process.terminate()
            try:
                process.wait(timeout)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(2.0)
    except psutil.TimeoutExpired as e:
        raise ProcessTerminationError(f"Termination timeout: {e}") from e


@contextmanager
def managed_process(
    executable: Executable, args: Optional[list[str]] = None, force: bool = False, timeout: float = 5.0
) -> Generator[psutil.Process, None, None]:
    """Context manager for process lifecycle."""
    process: psutil.Process = spawn_process(executable, args)
    try:
        yield process
    finally:
        terminate_process(process, force, timeout)
