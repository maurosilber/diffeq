from __future__ import annotations

import numpy as np

from numba import float64, njit
from numba.experimental import jitclass

from ..protocol import NDArray
from .utils import RHS, TRANSFORM


@njit
def identity(t: float, y: NDArray, p: NDArray):
    return y


@jitclass(
    [
        ("rhs", RHS.as_type()),
        ("y", float64[::1]),
        ("p", float64[::1]),
        ("transform", TRANSFORM.as_type()),
    ]
)
class ODEProblem:
    t: float
    y: np.ndarray
    p: np.ndarray

    def __init__(
        self,
        rhs,
        t: float,
        y: np.ndarray,
        p: np.ndarray = np.zeros(0),
        transform=identity,
    ):
        self.rhs = rhs
        self.t = t
        self.y = y
        self.p = p
        self.transform = transform
