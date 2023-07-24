from __future__ import annotations

from numba import float64
from numba.experimental import jitclass

from ..protocol import NDArray


@jitclass(
    [
        ("t", float64[::1]),
        ("y", float64[:, ::1]),
    ]
)
class Solution:
    t: NDArray
    y: NDArray

    def __init__(self, t: NDArray, y: NDArray):
        self.t = t
        self.y = y

    def eval(self, t: float):
        raise NotImplementedError
