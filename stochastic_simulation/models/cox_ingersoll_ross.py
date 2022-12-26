import numpy as np
from numpy.typing import ArrayLike
import scipy.stats as sps

from stochastic_simulation.models.base_model \
        import BaseModelDifferentiableTwice, args2arrays_decorator
from stochastic_simulation.random_number_generator import BaseRNG


class CoxIngersollRoss(BaseModelDifferentiableTwice):
    """
    dX_t = theta * (beta - X_t)dt + sigma sqrt(X_t) dW_t
    """
    def __init__(self, x_0: float, theta: float, beta: float, sigma: float):
        super().__init__(x_0)
        self.th = theta
        self.b = beta
        self.sig = sigma

    def get_generator(self):
        return CIRRNG(self.x_0, self.th, self.b, self.sig)

    @args2arrays_decorator
    def rate(self, _t: ArrayLike, xt: ArrayLike):
        return self.th * (self.b - xt) # type: ignore

    def rate_x(self, _t, _xt):
        return -self.th

    def rate_xx(self, _t, _xt):
        return 0

    @args2arrays_decorator
    def sigma(self, _t: ArrayLike, xt: ArrayLike):
        return self.sig * np.sqrt(xt)

    @args2arrays_decorator
    def sigma_x(self, _t, xt: ArrayLike):
        return 0.5 * self.sig / np.sqrt(xt)

    @args2arrays_decorator
    def sigma_xx(self, _t, xt: ArrayLike):
        return -0.25 * self.sig / xt**1.5 # type: ignore


class CIRRNG(BaseRNG):
    def __init__(self, x_0, theta, beta, sigma):
        super().__init__(x_0)
        self.th = theta
        self.b = beta
        self.sig = sigma

    def __rng(self, x_prev: float, dt: float):
        c = 2 * self.th / ((1 - np.exp(-self.th * dt)) * self.sig ** 2)
        ncp = 2 * c * x_prev * np.exp(-self.th * dt)
        df = 4 * self.th * self.b / self.sig**2
        dist = sps.ncx2(df=df, nc=ncp)
        return c, dist

    def gen_next(self, x_prev: float, dt: float):
        c, rng = self.__rng(x_prev, dt)
        return 0.5 * rng.rvs() / c

    def cdf(self, x_prev: float, dt: float, x_grid: ArrayLike):
        c, rng = self.__rng(x_prev, dt)
        return rng.cdf(x_grid * 2 * c) # type: ignore

    def pdf(self, x_prev: float, dt: float, x_grid: ArrayLike):
        c, rng = self.__rng(x_prev, dt)
        return rng.pdf(x_grid * 2 * c) * 2 * c # type: ignore

    def ppf(self, x_prev: float, dt: float, x_grid: ArrayLike):
        c, rng = self.__rng(x_prev, dt)
        return 0.5 * rng.ppf(x_grid) / c # type: ignore
