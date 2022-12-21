from enum import Enum

import numpy as np
from scipy.stats import multivariate_normal

from stochastic_simulation.models.base_model \
        import BaseModel, BaseModelDifferentiable, BaseModelDifferentiableTwice
from stochastic_simulation.wiener import generate


class ApproximationMethod(Enum):
    Euler = "euler"
    MilsteinFirstOrder = "milstein1"
    MilsteinSecondOrder = "milstein2"


def euler_step(model: BaseModel, t: float, x: float, dt: float, dw: float):
    return x + model.rate(t, x) * dt + model.sigma(t, x) * dw

def milstein1_step(model: BaseModelDifferentiable,
                   t: float, x: float, dt: float, dw: float):
    x_new = x + model.rate(t, x) * dt + model.sigma(t, x) * dw \
            + 0.5 * model.sigma(t, x) * model.sigma_x(t, x) * (dw**2 - dt)
    return x_new

def milstein2_step(model: BaseModelDifferentiableTwice,
                   t: float, x: float, dt: float, dw: float):
    r, rx, rxx = model.rate(t, x), model.rate_x(t, x), model.rate_xx(t, x)
    s, sx, sxx = model.sigma(t, x), model.sigma_x(t, x), model.sigma_xx(t, x)
    x_new = x + (r - 0.5 * s * sx) * dt + s * dw + 0.5 * s * sx * dw**2 \
            + 0.5 * (r * sx + rx * s + 0.5 * s**2 * sxx) * dt * dw \
            + (0.5 * r * rx + 0.25 * rxx * s**2) * dt**2
    return x_new

def predictor_corrector_approximate(model: BaseModelDifferentiable,
                                    end_t: float,
                                    weights: tuple[float, float] = (0.5, 0.5),
                                    number_of_points: int = 100,
                                    wieners = None):
    if wieners is None:
        ts, wieners = generate(end_t, number_of_points)
    else:
        ts = np.linspace(0, end_t, number_of_points)
    alpha, eta = weights
    rate_corr = lambda t, x: model.rate(t, x) - eta * model.sigma(t, x) \
                                                    * model.sigma_x(t, x)
    xt = [model.x_0]
    for i, dw in zip(range(0, number_of_points-1), np.diff(wieners)):
        dt = ts[i+1] - ts[i]
        x_pred = xt[-1] + model.rate(ts[i], xt[-1]) * dt\
                        + model.sigma(ts[i], xt[-1]) * dw
        x_corr = xt[-1] + (alpha * rate_corr(ts[i+1], x_pred) \
                           + (1 - alpha) * rate_corr(ts[i], xt[-1])) * dt \
                        + (eta * model.sigma(ts[i+1], x_pred) \
                           + (1 - eta) * model.sigma(ts[i], xt[-1])) * dw
        xt.append(x_corr)
    xt = np.array(xt)
    return ts, xt

def kps_approximate(model: BaseModelDifferentiableTwice,
                    end_t: float,
                    number_of_points: int = 100):
    ts, dt = np.linspace(0, end_t, number_of_points, retstep=True)
    cov_matrix = ((dt, 0.5 * dt**2), (0.5 * dt**2, dt**3 / 3))
    ws, us = multivariate_normal(
            mean=(0, 0),
            cov=cov_matrix, # type: ignore
        ).rvs(number_of_points - 1).T
    xt = [model.x_0]
    for t, w, u in zip(ts[1:], ws, us):
        x = xt[-1]
        r, rx, rxx = model.rate(t, x), model.rate_x(t, x), model.rate_xx(t, x)
        s, sx, sxx = model.sigma(t, x), model.sigma_x(t, x), model.sigma_xx(t, x)
        x_new = x + r * dt + s * w + 0.5 * s * sx * (w**2 - dt) \
                + s * rx * u + 0.5 * (r * rx - 0.5 * s * rxx) * dt**2 \
                + (r * sx + 0.5 * s**2 * sxx) * (w * dt - u) \
                + 0.5 * s * (sx**2 + s * sxx) * (w**2 / 3 - dt) * w
        xt.append(x_new)
    xt = np.array(xt)
    return ts, xt


def get_step_function(method: ApproximationMethod):
    match method:
        case ApproximationMethod.Euler:
            approximation_step = euler_step
        case ApproximationMethod.MilsteinFirstOrder:
            approximation_step = milstein1_step
        case ApproximationMethod.MilsteinSecondOrder:
            approximation_step = milstein2_step
        case _:
            approximation_step = euler_step
    return approximation_step

def approximate(model: BaseModelDifferentiableTwice,
                end_t: float,
                number_of_points: int = 100,
                method: ApproximationMethod | str = "euler",
                wieners = None):
    method = ApproximationMethod(method)
    approximation_step = get_step_function(method)

    if wieners is None:
        ts, wieners = generate(end_t, number_of_points)
    else:
        ts = np.linspace(0, end_t, number_of_points)
    dt = end_t / number_of_points

    xt = [model.x_0]
    for t, dw in zip(ts[1:], np.diff(wieners)):
        x = xt[-1]
        x_new = approximation_step(model, t, x, dt, dw)
        xt.append(x_new)
    xt = np.array(xt)
    return ts, xt
