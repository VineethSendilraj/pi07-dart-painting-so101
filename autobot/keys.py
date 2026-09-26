"""Single-key terminal input without Enter (macOS/Linux)."""

import select
import sys
import termios
import tty
from contextlib import contextmanager


@contextmanager
def keyboard():
    """Yields a non-blocking getter: returns one pressed character, or None."""
    if not sys.stdin.isatty():
        yield lambda: None
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield lambda: sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
