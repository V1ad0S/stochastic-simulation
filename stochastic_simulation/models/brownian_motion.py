import numpy as np
from numpy.typing import ArrayLike

from stochastic_simulation.wiener import generate
from stochastic_simulation.models.base_model \
        import BaseModelDifferentiableTwice, args2arrays_decorator


class BrownianMotion(BaseModelDifferentiableTwice):
    """
    dX_t = theta_1 * X_t * dt + theta_2 * X_t dW_t
    """
    def __init__(self, x_0: float, theta_1: float, theta_2: float):
        super().__init__(x_0)
        self.th1 = theta_1
        self.th2 = theta_2

    def generate(self, end_t: float, number_of_points: int = 100,
                 number_of_paths: int = 1, wieners = None):
        if wieners is None:
            ts, wieners = generate(end_t, number_of_points, number_of_paths)
        else:
            ts = np.linspace(0, end_t, number_of_points)
        xt = self.x_0 * np.exp(
                (self.th1 - self.th2**2 / 2) * ts + self.th2 * wieners
            )
        return ts, xt

    @args2arrays_decorator
    def rate(self, _t: ArrayLike, xt: ArrayLike):
        return self.th1 * xt # type: ignore

    def rate_x(self, _t: ArrayLike, _xt: ArrayLike):
        return self.th1

    def rate_xx(self, _t: ArrayLike, _xt: ArrayLike):
        return 0

    @args2arrays_decorator
    def sigma(self, _t: ArrayLike, xt: ArrayLike):
        return self.th2 * xt # type: ignore

    def sigma_x(self, _t: ArrayLike, _xt: ArrayLike):
        return self.th2

    def sigma_xx(self, _t: ArrayLike, _xt: ArrayLike):
        return 0
