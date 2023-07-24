from __future__ import annotations

import typing as ty

import numpy as np

from numba import float64
from numba.experimental import jitclass
from numba.typed import List

from ..protocol import Problem, Solver
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
        t = solver.t[-1]
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
