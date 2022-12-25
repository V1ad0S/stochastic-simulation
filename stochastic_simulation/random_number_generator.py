import numpy as np
from abc import ABC, abstractmethod



class BaseRNG(ABC):
    def __init__(self, x_0: float):
        self.x_0 = x_0

    @abstractmethod
    def gen_next(self, x_prev: float, dt: float):
        raise NotImplementedError

    @abstractmethod
    def cdf(self, x_prev: float, dt: float, x_grid):
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x_prev: float, dt: float, x_grid):
        raise NotImplementedError

    @abstractmethod
    def ppf(self, x_prev: float, dt: float, x_grid):
        raise NotImplementedError


def generate_from_rng(rng: BaseRNG, end_t: float, number_of_points: int = 100):
    ts, dt = np.linspace(0, end_t, number_of_points, retstep=True)
    xt = [rng.x_0]
    for _ in range(number_of_points - 1):
        xt.append(rng.gen_next(xt[-1], dt)) # type: ignore
    xt = np.array(xt)
    return ts, xt
