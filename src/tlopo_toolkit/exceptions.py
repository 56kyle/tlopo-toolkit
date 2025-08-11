

class TLOPOError(Exception):
    """Base class for exceptions in the tlopo_toolkit package."""


class ProcessSpawnError(TLOPOError):
    """Raised when process spawning fails."""
    pass


class ProcessTerminationError(TLOPOError):
    """Raised when process termination fails."""
    pass


class WindowNotFoundError(TLOPOError):
    """Raised when window cannot be found for process."""
    pass
