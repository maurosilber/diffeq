from __future__ import annotations

from dataclasses import dataclass

from ..protocol import NDArray, Solution


@dataclass
class Solution(Solution):
    t: NDArray
    y: NDArray

    def eval(self, t: float):
        raise NotImplementedError
