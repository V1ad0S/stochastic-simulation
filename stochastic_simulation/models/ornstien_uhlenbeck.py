import numpy as np
from numpy.typing import ArrayLike

from stochastic_simulation.wiener import generate
from stochastic_simulation.models.base_model \
        import BaseModelDifferentiable, args2arrays_decorator


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
