import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil

from tlopo_toolkit.exceptions import ProcessSpawnError
from tlopo_toolkit.exceptions import ProcessTerminationError


@dataclass(frozen=True)
class ProcessInfo:
    """Simple PID container."""
    pid: int



class Process:
    """Manages the lifecycle of a single executable instance."""

    def __init__(self, executable_path: Path, working_directory: Optional[Path] = None) -> None:
        self.executable_path = executable_path
        self.working_directory = working_directory or executable_path.parent
        self._pid: Optional[int] = None

    @property
    def is_running(self) -> bool:
        """Check if this executable instance is currently running."""
        if self._pid is None:
            return False
        try:
            process = psutil.Process(self._pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    @property
    def process_info(self) -> Optional[ProcessInfo]:
        """Get process info if running."""
        return ProcessInfo(pid=self._pid) if self._pid else None

    def spawn(self, command_args: list[str] = None) -> ProcessInfo:
        """Spawn the executable with given command arguments."""
        if self.is_running:
            raise ProcessSpawnError(f"Executable already running with PID {self._pid}")

        command_args = command_args or []
        full_command = [str(self.executable_path)] + command_args

        try:
            process = subprocess.Popen(
                full_command,
                cwd=str(self.working_directory),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            self._pid = process.pid
            return ProcessInfo(pid=process.pid)
        except (OSError, subprocess.SubprocessError) as e:
            raise ProcessSpawnError(f"Failed to spawn executable: {e}") from e

    def terminate(self, force: bool = False, timeout: float = 5.0) -> None:
        """Terminate the running executable."""
        if not self.is_running:
            self._pid = None
            return

        try:
            process = psutil.Process(self._pid)
            if not force:
                process.terminate()
                try:
                    process.wait(timeout=timeout)
                except psutil.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
            else:
                process.kill()
                process.wait(timeout=timeout)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        except psutil.TimeoutExpired as e:
            raise ProcessTerminationError(f"Executable termination timed out after {timeout}s") from e
        finally:
            self._pid = None
