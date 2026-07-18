"""Random-number helpers ported from the MATLAB algorithms.

Everything draws from a passed-in ``numpy.random.Generator`` so that each
parallel run is fully reproducible from its seed.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import levy_stable, cauchy, norm


def laprnd(rng: np.random.Generator, m: int = 1, n: int = 1,
           mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """i.i.d. Laplacian samples (port of ``laprnd``)."""
    u = rng.random((m, n)) - 0.5
    b = sigma / np.sqrt(2.0)
    return mu - b * np.sign(u) * np.log(1.0 - 2.0 * np.abs(u))


def lnf2(rng: np.random.Generator, alpha: float, scale: float,
         m: int = 1, n: int = 1) -> np.ndarray:
    """Port of ``LnF2`` (Linnik-type generator used by the DCS family).

    xhold = laprnd; SE = sign(rand-0.5)*xhold; U = rand;
    xhold = (sin(0.5*pi*a)*cot(0.5*pi*(a*U)) - cos(0.5*pi*a))^(1/a);
    out   = scale * SE / xhold
    """
    xhold = laprnd(rng, m, n, 0.0, 1.0)
    se = np.sign(rng.random((m, n)) - 0.5) * xhold
    u = rng.random((m, n))
    cot = 1.0 / np.tan(0.5 * np.pi * (alpha * u))
    denom = (np.sin(0.5 * np.pi * alpha) * cot
             - np.cos(0.5 * np.pi * alpha)) ** (1.0 / alpha)
    return scale * se / denom


def lnf2_scalar(rng: np.random.Generator, alpha: float, scale: float) -> float:
    return float(lnf2(rng, alpha, scale, 1, 1)[0, 0])


def dba_levy(rng: np.random.Generator, beta: float = 1.5, size=None):
    """Levy step(s) used inside ``DBA_d``.

    Faithful (including the original ``normrnd(0, sigma1^2)`` quirk where the
    second argument is treated by MATLAB as a standard deviation).  Pass
    ``size`` to draw a whole vector at once.
    """
    from math import gamma, sin, pi
    sigma1 = (gamma(1 + beta) * sin(pi * beta / 2.0)
              / (beta * gamma(0.5 + 0.5 * beta) * 2 ** (0.5 * beta - 0.5)))
    numerator = rng.normal(0.0, sigma1 ** 2, size=size)
    denom = np.abs(rng.normal(0.0, 1.0, size=size)) ** (-beta)
    return numerator / denom


# --------------------------------------------------------------------------- #
# COLSHADE stable distributions                                               #
# --------------------------------------------------------------------------- #
def normal_rng(rng: np.random.Generator, size=None) -> np.ndarray:
    """makedist('Normal', mu=0, sigma=0.1)."""
    return norm.rvs(loc=0.0, scale=0.1, size=size, random_state=rng)


def cauchy_rng(rng: np.random.Generator, size=None) -> np.ndarray:
    """makedist('Stable', alpha=1, beta=0, gam=0.1, delta=0) == Cauchy(0, 0.1)."""
    return cauchy.rvs(loc=0.0, scale=0.1, size=size, random_state=rng)


def levy_rng(rng: np.random.Generator, size=None) -> np.ndarray:
    """makedist('Stable', alpha=0.5, beta=1, gam=0.01, delta=0.01)."""
    return levy_stable.rvs(0.5, 1.0, loc=0.01, scale=0.01,
                           size=size, random_state=rng)
