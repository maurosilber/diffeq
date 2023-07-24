from dataclasses import dataclass

import numpy as np

from ..protocol import Problem, Solver
from . import cache


@dataclass
class Euler(Solver):
    dt: float

    def init(self, problem: Problem):
        self = self.__class__(self.dt)

        # Internal state
        self.rhs = problem.rhs
        self.p = problem.p
        CACHE_SIZE = 2
        self.t = cache.Number(CACHE_SIZE, problem.t)
        self.y = cache.Array(CACHE_SIZE, problem.y)
        self.dy = np.zeros_like(problem.y, dtype=float)
        return self

    def step(self) -> float:
        t = self.t.value
        y = self.y.value

        # update dy
        self.rhs(t, y, self.p, self.dy)

        # step
        t = t + self.dt
        y = y + self.dt * self.dy

        # update internal state
        self.t.append(t)
        self.y.append(y)
        return t

    def interpolate(self, t: float):
        """Linear interpolation."""
        if not self.t.values[0] <= t <= self.t.values[1]:
            raise ValueError

        slope = (self.y.values[1] - self.y.values[0]) / (
            self.t.values[1] - self.t.values[0]
        )
        return self.y.values[0] + slope * (t - self.t.values[0])
