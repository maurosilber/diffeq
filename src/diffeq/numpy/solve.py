from __future__ import annotations

from ..protocol import Problem, Saver, Solution, Solver


def solve(
    problem: Problem,
    *,
    t_end: float,
    solver: Solver,
    saver: Saver,
) -> Solution:
    solver = solver.init(problem)
    saver = saver.init(problem)
    t = problem.t
    while t < t_end:
        t = solver.step()
        saver.save(problem, solver)
    return saver.to_solution()
