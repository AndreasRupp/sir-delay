"""Estimation routines: exponential growth rate and piecewise-constant beta(t)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .model import solve

__all__ = ["GrowthFit", "fit_growth_rate", "BetaFit", "fit_piecewise_beta",
           "profile_changepoint"]


@dataclass
class GrowthFit:
    r: float
    r_stderr: float
    r_lo: float
    r_hi: float
    intercept: float
    r_squared: float
    n: int

    def __str__(self) -> str:
        return (f"r = {self.r:.4f} /day  (95% CI {self.r_lo:.4f}-{self.r_hi:.4f}), "
                f"R^2 = {self.r_squared:.4f}, n = {self.n}")


def fit_growth_rate(t, y) -> GrowthFit:
    """Ordinary least squares of ``log y`` on ``t``.

    Points with non-positive ``y`` are dropped.  The reported interval is the
    usual 95% Wald interval for the slope, which ignores the strong serial
    correlation of epidemic case counts and is therefore optimistic; it is
    quoted only to indicate the sampling precision of the window.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y = t[keep], np.log(y[keep])
    n = len(t)
    if n < 3:
        raise ValueError("need at least three positive observations")
    A = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = n - 2
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = float(np.sqrt(cov[1, 1]))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    from scipy.stats import t as tdist
    crit = float(tdist.ppf(0.975, dof))
    return GrowthFit(r=float(coef[1]), r_stderr=se, r_lo=float(coef[1] - crit * se),
                     r_hi=float(coef[1] + crit * se), intercept=float(coef[0]),
                     r_squared=r2, n=n)


# ---------------------------------------------------------------------------
# piecewise-constant transmission rate
# ---------------------------------------------------------------------------

@dataclass
class BetaFit:
    beta1: float
    beta2: float
    t0: float
    tau: float
    beta_mode: str
    sse: float
    rmse: float
    reduction: float          # 1 - beta2/beta1
    model_t: np.ndarray
    model_I: np.ndarray


def _simulate(par, *, beta1, beta2, t0, tau, beta_mode, hist, T, h, t_eval,
              observable="incidence"):
    """Run the model with a step transmission rate and sample it at ``t_eval``.

    ``observable`` selects what is compared against data: ``"I"`` is the
    prevalence, ``"incidence"`` the infection term beta*S(t-tau)*I(t-tau), which
    is what a daily case count reports.
    """
    def beta_fn(t):
        return beta1 if t < t0 else beta2

    sol = solve(
        Lambda=par.Lambda, beta=beta1, gamma=par.gamma,
        mu_S=par.mu_S, mu_I=par.mu_I, mu_R=par.mu_R, tau=tau,
        S0=hist["S"], I0=hist["I"], R0=hist["R"],
        T=T, h=h, beta_fn=beta_fn, beta_mode=beta_mode,
        history_fn=hist.get("fn"),
    )
    series = sol.incidence if observable == "incidence" else sol.I
    return sol, np.interp(t_eval, sol.t, np.asarray(series))


def fit_piecewise_beta(par, *, t_data, y_data, hist, tau, t0, beta_mode="current",
                       observable="incidence", h=0.05, T=None,
                       x0=(0.4, 0.1)) -> BetaFit:
    """Fit ``beta1`` (before ``t0``) and ``beta2`` (after) to observed data.

    The residual is taken on the log scale so that the pre-peak and post-peak
    parts of the curve carry comparable weight; ``t0`` is held fixed here and
    scanned separately by :func:`profile_changepoint`.
    """
    t_data = np.asarray(t_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    T = float(t_data.max()) if T is None else T
    log_obs = np.log(y_data)

    def residual(x):
        b1, b2 = np.exp(x)      # positivity of the rates by construction
        _, y = _simulate(par, beta1=b1, beta2=b2, t0=t0, tau=tau,
                         beta_mode=beta_mode, hist=hist, T=T, h=h,
                         t_eval=t_data, observable=observable)
        return np.log(np.maximum(y, 1e-300)) - log_obs

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = least_squares(residual, np.log(np.asarray(x0, dtype=float)),
                            method="trf", xtol=1e-12, ftol=1e-12)
        b1, b2 = np.exp(out.x)
        sol, y = _simulate(par, beta1=b1, beta2=b2, t0=t0, tau=tau,
                           beta_mode=beta_mode, hist=hist, T=T, h=h,
                           t_eval=t_data, observable=observable)
    sse = float(out.fun @ out.fun)
    return BetaFit(beta1=float(b1), beta2=float(b2), t0=float(t0), tau=float(tau),
                   beta_mode=beta_mode, sse=sse,
                   rmse=float(np.sqrt(sse / len(t_data))),
                   reduction=float(1.0 - b2 / b1),
                   model_t=sol.t,
                   model_I=np.asarray(sol.incidence if observable == "incidence"
                                      else sol.I))


def profile_changepoint(par, *, t_data, y_data, hist, tau, t0_grid,
                        beta_mode="current", observable="incidence",
                        h=0.05, T=None, x0=(0.4, 0.1)):
    """Profile the fit over a grid of change-point times.

    Returns ``(t0_grid, sse, best_fit)``.  Scanning rather than optimising over
    ``t0`` is deliberate: the residual is only piecewise smooth in ``t0`` (the
    transmission rate jumps), and the profile curve itself is informative -- its
    minimiser is the change-point the model *infers* from the data, which is the
    quantity compared against the known intervention date.
    """
    sse, fits = [], []
    for t0 in t0_grid:
        fit = fit_piecewise_beta(par, t_data=t_data, y_data=y_data, hist=hist,
                                 tau=tau, t0=float(t0), beta_mode=beta_mode,
                                 observable=observable, h=h, T=T, x0=x0)
        sse.append(fit.sse)
        fits.append(fit)
    sse = np.asarray(sse)
    return np.asarray(t0_grid, dtype=float), sse, fits[int(np.argmin(sse))]
