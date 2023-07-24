from __future__ import annotations

import numpy as np

from numba import float64
from numba.experimental import jitclass


@jitclass
class Number:
    values: float64[::1]

    def __init__(self, cache_size: int, value: float):
        self.values = self.values = np.empty(cache_size)
        self.values[-1] = value

    @property
    def value(self):
        return self.values[-1]

    def append(self, value):
        self.values[:-1] = self.values[1:]
        self.values[-1] = value


@jitclass
class Array:
    values: float64[:, ::1]

    def __init__(self, cache_size: int, value):
        self.values = np.empty((cache_size, len(value)))
        self.values[-1] = value

    @property
    def value(self):
        return self.values[-1]

    def append(self, value):
        self.values[:-1] = self.values[1:]
        self.values[-1] = value
