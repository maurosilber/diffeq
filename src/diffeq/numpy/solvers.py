from dataclasses import dataclass

import numpy as np

from ..protocol import Problem, Solver


@dataclass
class Euler(Solver):
    dt: float

    def init(self, problem: Problem):
        self = self.__class__(self.dt)

        # Internal state
        self.rhs = problem.rhs
        self.p = problem.p
        CACHE_SIZE = 2
        self.t = np.full(CACHE_SIZE, problem.t, dtype=float)
        self.y = np.full((CACHE_SIZE, len(problem.y)), problem.y, dtype=float)
        self.dy = np.zeros_like(problem.y, dtype=float)
        return self

    def step(self) -> float:
        t = self.t[-1]
        y = self.y[-1]

        # update dy
        self.rhs(t, y, self.p, self.dy)

        # step
        t = t + self.dt
        y = y + self.dt * self.dy

        # update internal state
        self.t[0] = self.t[-1]
        self.t[1] = t
        self.y[0] = self.y[-1]
        self.y[1] = y
        return t

    def interpolate(self, t: float):
        """Linear interpolation."""
        if not self.t[0] <= t <= self.t[1]:
            raise ValueError

        slope = (self.y[1] - self.y[0]) / (self.t[1] - self.t[0])
        return self.y[0] + slope * (t - self.t[0])
