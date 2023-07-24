from numpy.typing import NDArray


def copy_if_shared(x: NDArray):
    if x.base is None:
        return x
    else:
        return x.copy()
