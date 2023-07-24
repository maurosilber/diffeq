from __future__ import annotations

import typing as ty

import numpy as np

from numba import float64
from numba.experimental import jitclass
from numba.typed import List

from ..protocol import NDArray, Problem, Solver
from .solution import Solution


@jitclass
class AllSteps:
    t: ty.List[float64]
    y: ty.List[float64[::1]]

    def __init__(self):
        self.t = List([0.0])
        self.y = List([np.zeros(0)])

    def init(self, problem: Problem):
        self = AllSteps()
        t = problem.t
        y = problem.y
        p = problem.p
        self.t[0] = t
        self.y[0] = problem.transform(t, y, p)
        return self

    def save(self, problem: Problem, solver: Solver):
        t = solver.t.value
        y = solver.interpolate(t)
        p = problem.p
        self.t.append(t)
        self.y.append(problem.transform(t, y, p))

    def to_solution(self):
        N = len(self.t)
        t = np.empty(N)
        for i in range(N):
            t[i] = self.t[i]
        y = np.empty((N, len(self.y[0])))
        for i in range(N):
            y[i] = self.y[i]
        return Solution(t, y)


@jitclass
class AtTimes:
    i: int
    t: float64[::1]
    y: float64[:, ::1]

    def __init__(self, t: NDArray):
        self.i = 0
        self.t = t

    def init(self, problem: Problem):
        if problem.t > self.t[0]:
            raise ValueError

        self = AtTimes(self.t.copy())

        t = problem.t
        y = problem.y
        p = problem.p

        y = problem.transform(t, y, p)
        self.y = np.empty((len(self.t), len(y)))
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
        return Solution(self.t, self.y)
