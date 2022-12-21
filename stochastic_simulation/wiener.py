import numpy as np
import scipy.stats as sps


def generate(end_t: float,
             number_of_points: int = 100,
             number_of_paths: int = 1) -> tuple[np.ndarray, np.ndarray]:
    time_grid, dt = np.linspace(0, end_t, number_of_points, retstep=True)
    normal_sample = sps.norm(scale=dt**0.5).rvs((number_of_paths, number_of_points))
    normal_sample[:, 0] = 0
    wiener = normal_sample.cumsum(axis=1)
    if number_of_paths == 1:
        return time_grid, wiener[0]
    return time_grid, wiener
