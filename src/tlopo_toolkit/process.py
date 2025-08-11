import asyncio
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil
from pydantic import BaseModel, field_validator

from tlopo_toolkit.exceptions import ProcessSpawnError
from tlopo_toolkit.exceptions import ProcessTerminationError


class ExecutableConfiguration(BaseModel):
    """Configuration for an executable that can be launched."""
    executable_path: Path
    working_directory: Optional[Path] = None

    @classmethod
    @field_validator('executable_path')
    def validate_executable_exists(cls, path: Path) -> Path:
        if not path.exists():
            raise ValueError(f"Executable does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path

    def model_post_init(self, __context) -> None:
        if self.working_directory is None:
            object.__setattr__(self, 'working_directory', self.executable_path.parent)

    class Config:
        frozen = True


@dataclass(frozen=True)
class RunningProcess:
    """Represents an active process instance."""
    pid: int
    configuration: ExecutableConfiguration
    _subprocess_handle: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)

    def is_alive(self) -> bool:
        """Check if this process is currently running."""
        try:
            process = psutil.Process(self.pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False


@dataclass(frozen=True)
class ProcessInfo:
    """Legacy compatibility - simple PID container."""
    pid: int



class Process:
    """Manages the lifecycle of a single executable instance."""

    def __init__(self, executable_path: Path, working_directory: Optional[Path] = None) -> None:
        """Initialize with executable configuration."""
        self._configuration = ExecutableConfiguration(
            executable_path=executable_path,
            working_directory=working_directory
        )
        self._running_process: Optional[RunningProcess] = None

    @property
    def configuration(self) -> ExecutableConfiguration:
        """Get the executable configuration."""
        return self._configuration

    @property
    def running_process(self) -> Optional[RunningProcess]:
        """Get the currently running process, if any."""
        return self._running_process

    @property
    def is_running(self) -> bool:
        """Check if this executable instance is currently running."""
        return self._running_process is not None and self._running_process.is_alive()

    @property
    def process_info(self) -> Optional[ProcessInfo]:
        """Get legacy process info for compatibility."""
        return ProcessInfo(pid=self._running_process.pid) if self._running_process else None


    async def spawn_async(self, command_args: list[str]) -> ProcessInfo:
        """Spawn the executable asynchronously with given command arguments."""
        if self.is_running:
            raise ProcessSpawnError(f"Executable already running with PID {self._running_process.pid}")

        full_command = [str(self._configuration.executable_path)] + command_args

        try:
            subprocess_handle = await asyncio.create_subprocess_exec(
                *full_command,
                cwd=str(self._configuration.working_directory),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE
            )

            self._running_process = RunningProcess(
                pid=subprocess_handle.pid,
                configuration=self._configuration,
                _subprocess_handle=subprocess_handle
            )
            return ProcessInfo(pid=subprocess_handle.pid)

        except (OSError, subprocess.SubprocessError) as e:
            raise ProcessSpawnError(f"Failed to spawn executable: {e}") from e

    def spawn_sync(self, command_args: list[str]) -> ProcessInfo:
        """Spawn the executable synchronously with given command arguments."""
        if self.is_running:
            raise ProcessSpawnError(f"Executable already running with PID {self._running_process.pid}")

        full_command = [str(self._configuration.executable_path)] + command_args

        try:
            process = subprocess.Popen(
                full_command,
                cwd=str(self._configuration.working_directory),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )

            self._running_process = RunningProcess(
                pid=process.pid,
                configuration=self._configuration
            )
            return ProcessInfo(pid=process.pid)

        except (OSError, subprocess.SubprocessError) as e:
            raise ProcessSpawnError(f"Failed to spawn executable: {e}") from e

    async def terminate_async(self, force: bool = False, timeout: float = 5.0) -> None:
        """Terminate the running executable asynchronously with optional force and timeout."""
        if not self.is_running:
            self._clear_running_process()
            return

        try:
            process = psutil.Process(self._running_process.pid)

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
            self._clear_running_process()

    def terminate_sync(self, force: bool = False, timeout: float = 5.0) -> None:
        """Terminate the running executable synchronously with optional force and timeout."""
        if not self.is_running:
            self._clear_running_process()
            return

        try:
            process = psutil.Process(self._running_process.pid)

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
            self._clear_running_process()

    def _clear_running_process(self) -> None:
        """Clear the running process reference after termination."""
        self._running_process = None


