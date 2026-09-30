"""Linearised analysis and trajectory diagnostics.

The central object is the characteristic equation of the delayed SIR model at
the disease-free state.  Linearising

    I'(t) = beta * S(t-tau) * I(t-tau) - (gamma + mu_I) * I(t)

about (S*, 0, 0) with S* = Lambda / mu_S gives i'(t) = beta*S* * i(t-tau) -
(gamma + mu_I) * i(t) and hence

    r + gamma + mu_I = beta * S* * exp(-r * tau).                         (*)

Two consequences are used throughout the experiments:

* forward:  given beta, the early exponential growth rate r is the unique real
  root of (*), and it is strictly decreasing in tau even though R0 does not
  depend on tau at all;
* inverse:  given an *observed* growth rate r, the transmission rate required to
  reproduce it is beta = (r + gamma + mu_I) * exp(r*tau) / S*, so that

      R0_delay(tau) = (1 + r/(gamma+mu_I)) * exp(r*tau) = R0_ode * exp(r*tau).

The factor exp(r*tau) is the delta-kernel case of the Wallinga-Lipsitch relation
between growth rate and reproduction number: with a latent period tau, a share
of the generation interval is concentrated at a fixed lag, and ignoring it
biases R0 downwards.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "growth_rate", "beta_from_growth", "R0_from_growth", "R0_ode_from_growth",
    "peak_metrics", "history_sup",
]


def growth_rate(*, beta, gamma, mu_I, S_star, tau) -> float:
    """Unique real root r of  r + gamma + mu_I = beta*S* * exp(-r*tau).

    The left-hand side minus the right-hand side is strictly increasing in r
    (derivative 1 + tau*beta*S*exp(-r tau) > 0), so the real root is unique; for
    this scalar delay equation it is also the rightmost root of the
    characteristic function, hence the observed exponential growth rate.
    """
    a = beta * S_star
    c = gamma + mu_I
    if tau == 0:
        return a - c

    def f(r):
        return r + c - a * np.exp(-r * tau)

    # f(-c) = -a*exp(c*tau) < 0 and f grows without bound; bracket upwards.
    lo, hi = -c, max(1.0, a)
    while f(hi) < 0:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("failed to bracket the characteristic root")
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=1e-15))


def beta_from_growth(*, r, gamma, mu_I, S_star, tau):
    """Transmission rate reproducing an observed growth rate ``r`` at delay ``tau``."""
    return (r + gamma + mu_I) * np.exp(np.asarray(r) * np.asarray(tau)) / S_star


def R0_ode_from_growth(*, r, gamma, mu_I):
    """Delay-free estimate R0 = 1 + r/(gamma + mu_I)."""
    return 1.0 + np.asarray(r) / (gamma + mu_I)


def R0_from_growth(*, r, gamma, mu_I, tau):
    """Delay-model estimate R0 = (1 + r/(gamma+mu_I)) * exp(r*tau)."""
    return R0_ode_from_growth(r=r, gamma=gamma, mu_I=mu_I) * np.exp(
        np.asarray(r) * np.asarray(tau))


def peak_metrics(sol, *, warmup: float = 0.0) -> dict:
    """Peak time/height of I and cumulative incidence of a scalar run."""
    k0 = int(round(warmup / sol.h))
    I = np.asarray(sol.I)[k0:]
    t = sol.t[k0:]
    k = int(np.argmax(I))
    cum = float(np.sum(np.asarray(sol.incidence)) * sol.h)
    return {
        "t_peak": float(t[k]),
        "I_peak": float(I[k]),
        "final_S": float(np.asarray(sol.S)[-1]),
        "final_I": float(np.asarray(sol.I)[-1]),
        "final_R": float(np.asarray(sol.R)[-1]),
        "cumulative_incidence": cum,
    }


def history_sup(S0, I0, R0) -> float:
    """sup of N over the history for a constant history."""
    return float(np.asarray(S0) + np.asarray(I0) + np.asarray(R0))
