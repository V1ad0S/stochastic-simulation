import numpy as np
from numpy.typing import ArrayLike

from stochastic_simulation.models.base_model \
        import BaseModelDifferentiableTwice, args2arrays_decorator


class CoxIngersollRoss(BaseModelDifferentiableTwice):
    """
    dX_t = theta * (beta - X_t)dt + sigma sqrt(X_t) dW_t
    """
    def __init__(self, x_0: float, theta: float, beta: float, sigma: float):
        super().__init__(x_0)
        self.th = theta
        self.b = beta
        self.sig = sigma

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
