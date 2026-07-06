import numpy as np
import numba as nb

@nb.njit(cache=True, fastmath=True)
def ou_loop(theta: float, mu: float, shocks: np.ndarray, noise: np.ndarray):
    for t in range(1, noise.shape[0]):
        drift: float = theta * (mu - noise[t-1])
        noise[t] = noise[t - 1] + drift + shocks[t - 1]
        