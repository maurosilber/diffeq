import numpy as np
from pytest import mark

from .. import ODEProblem, savers, solve, solvers


@mark.parametrize(
    "saver",
    [
        savers.AllSteps(),
        savers.AtTimes(np.linspace(0, 1, 10)),
    ],
)
@mark.parametrize("solver", [solvers.Euler(dt=0.01)])
def test_solve(solver, saver):
    def rhs(t, y, p, dy):
        dy[:] = -y

    problem = ODEProblem(
        rhs,
        0.0,
        np.arange(5.0),
        np.zeros(0),
    )
    solution = solve(
        problem,
        t_end=1,
        solver=solver,
        saver=saver,
    )

    assert solution.t[0] == problem.t
    assert np.isclose(solution.t[-1], 1)
    assert len(solution.t) == len(solution.y)
    assert solution.y.shape[1] == problem.y.size
    assert np.allclose(
        solution.y,
        solution.y[:1] * np.exp(-solution.t[:, None]),
        atol=1e-2,
        rtol=1e-2,
    )
