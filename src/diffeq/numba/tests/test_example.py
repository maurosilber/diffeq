import numpy as np
from numba import njit

from .. import AllSteps, Euler, ODEProblem, solve


def test_solve():
    @njit
    def rhs(t, y, p, dy):
        dy[:] = -y

    problem = ODEProblem(
        rhs,
        0.0,
        np.arange(5.0),
    )
    solution = solve(
        problem,
        t_end=1,
        solver=Euler(dt=0.01),
        saver=AllSteps(),
    )

    assert solution.t[0] == problem.t
    assert solution.t[1] == 0.01
    assert np.isclose(solution.t[-1], 1)
    assert len(solution.t) == len(solution.y)
    assert solution.y.shape[1] == problem.y.size
