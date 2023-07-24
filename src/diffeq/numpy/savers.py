from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..protocol import NDArray, Problem, Saver, Solver
from .solution import Solution
from .utils import copy_if_shared


class AllSteps(Saver):
    t: list[float]
    y: list[NDArray]

    def init(self, problem: Problem):
        self = self.__class__()
        t = problem.t
        y = problem.y
        p = problem.p
        self.t = [t]
        self.y = [copy_if_shared(problem.transform(t, y, p))]
        return self

    def save(self, problem: Problem, solver: Solver):
        t = solver.t.value
        y = solver.y.value
        p = problem.p
        self.t.append(t)
        self.y.append(copy_if_shared(problem.transform(t, y, p)))

    def to_solution(self):
        return Solution(t=np.array(self.t), y=np.array(self.y))


@dataclass
class AtTimes(Saver):
    t: NDArray
    y: NDArray = field(init=False)

    def init(self, problem: Problem):
        if problem.t > self.t[0]:
            raise ValueError

        self = self.__class__(self.t)

        t = problem.t
        y = problem.y
        p = problem.p

        y = problem.transform(t, y, p)
        self.y = np.empty((len(self.t), len(y)))
        self.t = self.t.copy()
        if t == self.t[0]:
            self.y[0] = y
            self.i = 1
        else:
            self.i = 0
        return self

    def save(self, problem: Problem, solver: Solver):
        if self.i >= len(self.t):
            # Nothing more to do. Stop solver?
            return

        if solver.t.value >= self.t[self.i]:
            t = self.t[self.i]
            y = solver.interpolate(t)
            self.y[self.i] = problem.transform(t, y, problem.p)
            self.i += 1

    def to_solution(self):
        return Solution(t=self.t, y=self.y)
