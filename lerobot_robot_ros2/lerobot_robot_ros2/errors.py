"""Device connection errors for ROS2 plugins (lerobot 0.5+ removed lerobot.errors)."""


class DeviceNotConnectedError(RuntimeError):
    """Raised when an operation requires a connected device."""


class DeviceAlreadyConnectedError(RuntimeError):
    """Raised when connect() is called on an already connected device."""
