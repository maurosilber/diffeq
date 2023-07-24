from numba import float64, void

RHS = void(float64, float64[::1], float64[::1], float64[::1])
TRANSFORM = float64[::1](float64, float64[::1], float64[::1])
