import numpy as np
from numpy.typing import ArrayLike
import scipy.stats as sps

from stochastic_simulation.wiener import generate
from stochastic_simulation.models.base_model \
        import BaseModelDifferentiable, args2arrays_decorator
from stochastic_simulation.random_number_generator import BaseRNG


class OrnsteinUhlenbeck(BaseModelDifferentiable):
    """
    dX_t = theta * (mu - X_t)dt + sigma dW_t
    """
    def __init__(self, x_0: float, theta: float, mu: float, sigma: float):
        super().__init__(x_0)
        self.th = theta
        self.mu = mu
        self.sig = sigma

    def generate(self, end_t: float, number_of_points: int = 100,
                 number_of_paths: int = 1, wieners = None):
        if wieners is None:
            ts, wieners = generate(end_t, number_of_points, number_of_paths)
        else:
            ts = np.linspace(0, end_t, wieners.size)
        ito_sum = (np.exp(self.th * ts[1:]) * np.diff(wieners)).cumsum(axis=-1)
        exp = np.exp(-self.th * ts[1:])
        xt = self.x_0 * exp + self.mu * (1 - exp) + self.sig * exp * ito_sum
        xt = np.insert(xt, 0, self.x_0, axis=-1)
        return ts, xt

    def get_generator(self):
        return OURNG(self.x_0, self.th, self.mu, self.sig)

    @args2arrays_decorator
    def rate(self, _t: ArrayLike, xt: ArrayLike):
        return self.th * (self.mu - xt) # type: ignore

    def rate_x(self, _t, _xt):
        return -self.th

    def rate_xx(self, _t, _xt):
        return 0

    def sigma(self, _t, _xt):
        return self.sig

    def sigma_x(self, _t, _xt):
        return 0

    def sigma_xx(self, _t, _xt):
        return 0


class OURNG(BaseRNG):
    def __init__(self, x_0, theta, mu, sigma):
        super().__init__(x_0)
        self.th = theta
        self.mu = mu
        self.sig = sigma

    def __rng(self, x_prev: float, dt: float):
        mean = self.mu + (x_prev - self.mu) * np.exp(-self.th * dt)
        var = 0.5 * self.sig**2 * (1 - np.exp(-2 * self.th * dt)) / self.th
        return sps.norm(mean, var**0.5)

    def gen_next(self, x_prev: float, dt: float):
        return self.__rng(x_prev, dt).rvs()

    def cdf(self, x_prev: float, dt: float, x_grid):
        return self.__rng(x_prev, dt).cdf(x_grid)

    def pdf(self, x_prev: float, dt: float, x_grid):
        return self.__rng(x_prev, dt).pdf(x_grid) # type: ignore

    def ppf(self, x_prev: float, dt: float, x_grid):
        return self.__rng(x_prev, dt).ppf(x_grid)
