import numpy as np

from abc import ABC, abstractmethod
from numpy.typing import ArrayLike


def args2arrays_decorator(func):
    def wrapper(self, t: ArrayLike, xt: ArrayLike):
        t = np.array(t)
        xt = np.array(xt)
        return func(self, t, xt)
    return wrapper


class BaseModel(ABC):
    """
    dX_t = rate(t, X_t)dt + sigma(t, X_t)dW_t
    """
    def __init__(self, x_0: float):
        self.x_0 = x_0

    @abstractmethod
    def rate(self, t: ArrayLike, xt: ArrayLike):
        raise NotImplementedError

    @abstractmethod
    def sigma(self, t: ArrayLike, xt: ArrayLike):
        raise NotImplementedError


class BaseModelDifferentiable(BaseModel):
    @abstractmethod
    def sigma_x(self, t: ArrayLike, xt: ArrayLike):
        raise NotImplementedError


class BaseModelDifferentiableTwice(BaseModelDifferentiable):
    @abstractmethod
    def rate_x(self, t: ArrayLike, xt: ArrayLike):
        raise NotImplementedError

    @abstractmethod
    def rate_xx(self, t: ArrayLike, xt: ArrayLike):
        raise NotImplementedError

    @abstractmethod
    def sigma_xx(self, t: ArrayLike, xt: ArrayLike):
        raise NotImplementedError

__all__ = [
    "BaseModel",
    "BaseModelDifferentiable",
    "BaseModelDifferentiableTwice",
]
