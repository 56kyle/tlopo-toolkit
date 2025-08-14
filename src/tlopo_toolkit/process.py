import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil

from tlopo_toolkit.exceptions import ProcessSpawnError, ProcessTerminationError


@dataclass(frozen=True)
class ProcessInfo:
    """Process identification."""
    pid: int


class Process:
    """Simple process lifecycle management."""

    def __init__(self, executable_path: Path, working_directory: Optional[Path] = None):
        self.executable_path = executable_path
        self.working_directory = working_directory or executable_path.parent
        self._pid: Optional[int] = None

    @property
    def is_running(self) -> bool:
        """Check if process is running."""
        if not self._pid:
            return False
        try:
            return psutil.Process(self._pid).is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    @property
    def process_info(self) -> Optional[ProcessInfo]:
        """Get process info if running."""
        return ProcessInfo(self._pid) if self._pid else None

    def spawn(self, args: list[str] = None) -> ProcessInfo:
        """Spawn the process."""
        if self.is_running:
            raise ProcessSpawnError(f"Process already running (PID {self._pid})")

        cmd = [str(self.executable_path)] + (args or [])
        try:
            proc = subprocess.Popen(cmd, cwd=self.working_directory)
            self._pid = proc.pid
            return ProcessInfo(proc.pid)
        except (OSError, subprocess.SubprocessError) as e:
            raise ProcessSpawnError(f"Failed to spawn: {e}") from e

    def terminate(self, force: bool = False, timeout: float = 5.0) -> None:
        """Terminate the process."""
        if not self.is_running:
            self._pid = None
            return

        try:
            proc = psutil.Process(self._pid)
            if force:
                proc.kill()
            else:
                proc.terminate()
                try:
                    proc.wait(timeout)
                except psutil.TimeoutExpired:
                    proc.kill()
                    proc.wait(2.0)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        except psutil.TimeoutExpired as e:
            raise ProcessTerminationError(f"Termination timeout: {e}") from e
        finally:
            self._pid = None
