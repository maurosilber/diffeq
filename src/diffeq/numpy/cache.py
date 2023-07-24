from __future__ import annotations

import numpy as np

from ..protocol import Cache, NDArray, T


class Number(Cache[T]):
    values: NDArray

    def __init__(self, cache_size: int, value: T):
        self.values = self.values = np.empty(cache_size)
        self.values[-1] = value

    @property
    def value(self) -> T:
        return self.values[-1]

    def append(self, value: T):
        self.values[:-1] = self.values[1:]
        self.values[-1] = value


class Array(Cache):
    values: NDArray

    def __init__(self, cache_size: int, value: NDArray):
        self.values = np.empty((cache_size, len(value)))
        self.values[-1] = value

    @property
    def value(self) -> NDArray:
        return self.values[-1]

    def append(self, value: NDArray):
        self.values[:-1] = self.values[1:]
        self.values[-1] = value
