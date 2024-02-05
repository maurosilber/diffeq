from __future__ import annotations

from typing import Protocol, Self, TypeVar

from numpy.typing import NDArray

T = TypeVar("T")


class RHS(Protocol):
    """In-place right-hand side

    t: time
    y: state-vector
    p: parameter-vector
    dy: derivative of y, updated in-place
    """

    def __call__(
        self,
        t: float,
        y: NDArray,
        p: NDArray,
        dy: NDArray,
    ): ...


class Transform(Protocol):
    """Transformation of the solver state.

    t: time
    y: state-vector
    p: parameter-vector
    """

    def __call__(
        self,
        t: float,
        y: NDArray,
        p: NDArray,
    ) -> NDArray: ...


class Problem(Protocol):
    """Describes an initial value problem.

    rhs: right hand side (in-place)
    t: time
    y: state-vector
    p: parameter-vector
    transform: transformation of the state-vector to save
    """

    rhs: RHS
    t: float
    y: NDArray
    p: NDArray
    transform: Transform


class Cache(Protocol[T]):
    value: T
    values: NDArray

    def append(self, value: T): ...


class Solver(Protocol):
    t: Cache[float]
    y: Cache[NDArray]
    dy: NDArray

    def init(self, problem: Problem) -> Self:
        """Create the internal state from the given problem."""
        ...

    def step(self) -> float:
        """Move the internal state one step forward,
        return the new time."""
        ...

    def interpolate(self, t: float) -> NDArray:
        """Interpolate the solution self.y at the time t."""
        ...


class Saver(Protocol):
    def init(self, problem: Problem) -> Self: ...

    def save(self, problem: Problem, solver: Solver) -> None: ...

    def to_solution(self) -> Solution: ...


class Solution(Protocol):
    t: NDArray
    y: NDArray

    def eval(self, t: float) -> NDArray: ...
