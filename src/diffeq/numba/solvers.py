from __future__ import annotations

import numpy as np

from numba import float64
from numba.experimental import jitclass

from .utils import RHS


@jitclass(
    [
        ("rhs", RHS.as_type()),
        ("t", float64[::1]),
        ("y", float64[:, ::1]),
        ("p", float64[::1]),
        ("dy", float64[::1]),
    ]
)
class Euler:
    dt: float

    def __init__(self, dt: float):
        self.dt = dt

    def init(self, problem):
        self = Euler(self.dt)

        self.rhs = problem.rhs
        self.p = problem.p

        CACHE_SIZE = 2
        self.t = np.empty(CACHE_SIZE)
        self.y = np.empty((CACHE_SIZE, len(problem.y)))
        for i in range(CACHE_SIZE):
            self.t[i] = problem.t
            self.y[i] = problem.y
        self.dy = np.zeros(len(problem.y), float)
        return self

    def step(self) -> float:
        t = self.t[-1]
        y = self.y[-1]

        self.rhs(t, y, self.p, self.dy)
        t = t + self.dt
        y = y + self.dt * self.dy

        self.t[0] = self.t[-1]
        self.t[1] = t
        self.y[0] = self.y[-1]
        self.y[1] = y

        return t

    def interpolate(self, t: float):
        if not self.t[0] <= t <= self.t[1]:
            raise ValueError

        slope = (self.y[1] - self.y[0]) / (self.t[1] - self.t[0])
        return self.y[0] + slope * (t - self.t[0])
