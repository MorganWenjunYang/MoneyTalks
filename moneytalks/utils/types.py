"""Enumerations and type definitions for MoneyTalks."""

from enum import Enum


class Direction(Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE = "CLOSE"


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
