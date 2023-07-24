from __future__ import annotations

from dataclasses import dataclass

from ..protocol import RHS, NDArray, Problem, Transform


def identity(t: float, y: NDArray, p: NDArray):
    return y


@dataclass
class ODEProblem(Problem):
    rhs: RHS
    t: float
    y: NDArray
    p: NDArray
    transform: Transform = identity
