from __future__ import annotations

import numpy as np

from numba import float64
from numba.experimental import jitclass

from . import cache
from .utils import RHS


@jitclass(
    [
        ("rhs", RHS.as_type()),
        ("p", float64[::1]),
        ("dy", float64[::1]),
    ]
)
class Euler:
    dt: float
    t: cache.Number
    y: cache.Array

    def __init__(self, dt: float):
        self.dt = dt

    def init(self, problem):
        self = Euler(self.dt)

        self.rhs = problem.rhs
        self.p = problem.p

        CACHE_SIZE = 2
        self.t = cache.Number(CACHE_SIZE, problem.t)
        self.y = cache.Array(CACHE_SIZE, problem.y)
        self.dy = np.zeros(len(problem.y), float)
        return self

    def step(self) -> float:
        t = self.t.value
        y = self.y.value

        self.rhs(t, y, self.p, self.dy)
        t = t + self.dt
        y = y + self.dt * self.dy

        self.t.append(t)
        self.y.append(y)
        return t

    def interpolate(self, t: float):
        if not self.t.values[0] <= t <= self.t.values[1]:
            raise ValueError

        slope = (self.y.values[1] - self.y.values[0]) / (
            self.t.values[1] - self.t.values[0]
        )
        return self.y.values[0] + slope * (t - self.t.values[0])
