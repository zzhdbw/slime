"""Compatibility alias for the real tracking implementation.

The actual implementation lives in ``slime.observability.logging_utils`` because
that is what ``train.py``, train actors, and the rollout manager import.
"""

from slime.observability.logging_utils import (  # noqa: F401
    configure_logger,
    finish_tracking,
    init_tracking,
    log,
)

__all__ = [
    "configure_logger",
    "finish_tracking",
    "init_tracking",
    "log",
]
