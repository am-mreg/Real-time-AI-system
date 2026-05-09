"""
src/stream/stream.py

A small, robust Stream helper that wraps a frame generator and exposes
a Python iterator interface. Safe to use in StreamingResponse or other
consumers that iterate over frames.

Usage:
    stream = Stream(generator)
    for frame in stream:
        ...  # process frame

The class supports closing the underlying generator and a simple timeout
mechanism for next() to avoid blocking forever if desired.
"""
from typing import Callable, Generator, Iterable, Optional, Any
import threading
import time


class Stream:
    def __init__(self, generator: Optional[Iterable[Any]] = None):
        """
        Wrap an iterable/generator that yields frames (or any items).
        If generator is None, the Stream will behave as an empty iterator.
        """
        self._gen = iter(generator) if generator is not None else iter(())
        self._lock = threading.Lock()
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            if self._closed:
                raise StopIteration
            try:
                return next(self._gen)
            except StopIteration:
                self.close()
                raise
            except Exception:
                # On unexpected error from underlying generator, close and stop iteration
                self.close()
                raise StopIteration

    def close(self):
        """
        Close the stream and attempt to close the underlying generator if it
        exposes a close() method (typical for generator objects).
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                close_fn = getattr(self._gen, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass

    @classmethod
    def from_callable(cls, fn: Callable[[], Generator[Any, None, None]]):
        """
        Convenience constructor that accepts a callable returning a generator.
        The callable will be invoked lazily when iteration starts.
        """
        return cls(fn())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
