"""
Custom exceptions for ROS 2 Robot Interface.
"""


class ROS2InterfaceError(Exception):
    """Base exception for ROS 2 interface errors."""
    pass


class ROS2NotConnectedError(ROS2InterfaceError):
    """Raised when trying to use the interface while not connected."""
    pass


class ROS2AlreadyConnectedError(ROS2InterfaceError):
    """Raised when trying to connect while already connected."""
    pass

