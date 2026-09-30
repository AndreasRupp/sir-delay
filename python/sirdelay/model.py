"""Implicit-Euler solver for the delayed SIR model (Section 2.3 of the manuscript).

The scheme is the one stated in equation (6) of the manuscript.  Written out, the lower-triangular system solves to

    S^{n+1} = (S^n + h*Lambda - h*beta_n * S^{n+1-N} * I^{n+1-N}) / (1 + h*mu_S),
    I^{n+1} = (I^n            + h*beta_n * S^{n+1-N} * I^{n+1-N}) / (1 + h*gamma + h*mu_I),
    R^{n+1} = (R^n + h*gamma * I^{n+1})                           / (1 + h*mu_R),

with the step size chosen as h = tau / N so that the delay falls exactly on the
grid and no interpolation of the delayed term is needed.

Everything is vectorised over a trailing *sweep* axis: every coefficient may be
an array, in which case a whole family of parameter sets is integrated
simultaneously.  This is what makes the two-dimensional sharpness sweep of
experiment 1 (tens of thousands of parameter points) tractable in numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

__all__ = ["Solution", "solve", "grid_for"]


@dataclass
class Solution:
    """Result of a run.

    For ``record="full"`` the trajectory arrays have shape ``(n_steps+1,) + sweep``;
    for ``record="summary"`` they are ``None`` and only the running extrema are
    kept, which is what the large parameter sweeps use.
    """

    h: float
    n_delay: int
    t: np.ndarray
    S: Optional[np.ndarray]
    I: Optional[np.ndarray]
    R: Optional[np.ndarray]
    incidence: Optional[np.ndarray]
    min_S: np.ndarray
    min_I: np.ndarray
    min_R: np.ndarray
    max_N: np.ndarray
    t_min_S: np.ndarray

    @property
    def N(self) -> Optional[np.ndarray]:
        """Total population S + I + R."""
        if self.S is None:
            return None
        return self.S + self.I + self.R

    @property
    def positive(self) -> np.ndarray:
        """Boolean mask: did the discrete solution stay non-negative?"""
        return (self.min_S >= 0.0) & (self.min_I >= 0.0) & (self.min_R >= 0.0)

    def at(self, time: float) -> tuple:
        """Nearest-grid-point values of (S, I, R) at ``time``."""
        k = int(round(time / self.h))
        return self.S[k], self.I[k], self.R[k]


def grid_for(tau: float, h_target: float, T: float) -> tuple[float, int, int]:
    """Snap ``h_target`` to a step size with the delay on the grid.

    Returns ``(h, n_delay, n_steps)``.  For ``tau > 0`` the step is ``h = tau/N``
    with ``N = max(1, round(tau/h_target))``, exactly the manuscript's
    requirement.  For ``tau == 0`` the delay index is ``0`` and ``h_target`` is
    used unchanged, which reduces the scheme to semi-implicit Euler for the
    classical SIR model.
    """
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if tau == 0.0:
        h = float(h_target)
        n_delay = 0
    else:
        n_delay = max(1, int(round(tau / h_target)))
        h = tau / n_delay
    n_steps = max(1, int(round(T / h)))
    return h, n_delay, n_steps


def solve(
    *,
    Lambda,
    beta,
    gamma,
    mu_S,
    mu_I,
    mu_R,
    tau: float,
    S0,
    I0,
    R0,
    T: float,
    h: float,
    beta_fn: Optional[Callable[[float], object]] = None,
    beta_mode: str = "current",
    history_fn: Optional[Callable[[np.ndarray], tuple]] = None,
    record: str = "full",
) -> Solution:
    """Integrate the delayed SIR model with the manuscript's implicit-Euler scheme.

    Parameters
    ----------
    Lambda, beta, gamma, mu_S, mu_I, mu_R
        Model coefficients.  Scalars, or arrays that broadcast against each
        other to a common *sweep* shape.
    tau
        Constant delay.  ``tau = 0`` gives the delay-free SIR model.
    S0, I0, R0
        Constant history values on ``[-tau, 0]``.  Broadcast like the
        coefficients.  Ignored when ``history_fn`` is given.
    T, h
        Final time and target step size; ``h`` is snapped via :func:`grid_for`.
    beta_fn
        Optional time-dependent transmission rate, called with a scalar time and
        returning something broadcastable to the sweep shape.  Overrides ``beta``.
    beta_mode
        Which time the transmission rate is evaluated at when ``beta_fn`` is
        given.  ``"current"`` uses ``beta(t_{n+1})``, so that a step change in
        beta alters I' immediately; ``"delayed"`` uses ``beta(t_{n+1} - tau)``,
        i.e. the rate in force when the contact actually happened, so that the
        effect on I is postponed by tau.  The two readings are compared in
        experiment 3.
    history_fn
        Optional callable mapping an array of times in ``[-tau, 0]`` to a tuple
        ``(S, I, R)`` of arrays, for non-constant histories.
    record
        ``"full"`` stores the trajectories, ``"summary"`` only running extrema.
    """
    if beta_mode not in ("current", "delayed"):
        raise ValueError("beta_mode must be 'current' or 'delayed'")
    if record not in ("full", "summary"):
        raise ValueError("record must be 'full' or 'summary'")

    coeffs = np.broadcast_arrays(
        *(np.asarray(v, dtype=float)
          for v in (Lambda, beta, gamma, mu_S, mu_I, mu_R, S0, I0, R0))
    )
    sweep = coeffs[0].shape
    # Work with a flat trailing axis of length P internally; this keeps every
    # array at least one-dimensional and sidesteps 0-d/scalar special cases.
    # Outputs are reshaped back to ``sweep`` at the end.
    Lam, bet, gam, muS, muI, muR, s0, i0, r0 = (
        np.array(c, dtype=float).reshape(-1) for c in coeffs
    )
    flat = Lam.shape  # (P,)

    h, n_delay, n_steps = grid_for(tau, h, T)

    # Denominators of the triangular solve; constant in time.
    den_S = 1.0 + h * muS
    den_I = 1.0 + h * gam + h * muI
    den_R = 1.0 + h * muR

    # --- history -----------------------------------------------------------
    # Slot m % n_delay holds the value at time index m, so that the value needed
    # at step n, namely index n+1-N, sits in slot (n+1) % N -- the very slot the
    # new value is written to afterwards.
    if n_delay > 0:
        buf_S = np.empty((n_delay,) + flat, dtype=float)
        buf_I = np.empty((n_delay,) + flat, dtype=float)
        idx = np.arange(1 - n_delay, 1)          # time indices -N+1, ..., 0
        if history_fn is None:
            hist_S = np.broadcast_to(s0, (n_delay,) + flat)
            hist_I = np.broadcast_to(i0, (n_delay,) + flat)
        else:
            hs, hi, _ = history_fn(idx * h)
            hist_S = np.broadcast_to(
                np.asarray(hs, dtype=float).reshape(n_delay, -1), (n_delay,) + flat)
            hist_I = np.broadcast_to(
                np.asarray(hi, dtype=float).reshape(n_delay, -1), (n_delay,) + flat)
        for k, m in enumerate(idx):
            buf_S[m % n_delay] = hist_S[k]
            buf_I[m % n_delay] = hist_I[k]
        S = buf_S[0].copy()          # time index 0 lives in slot 0
        I = buf_I[0].copy()
    else:
        buf_S = buf_I = None
        S, I = s0.copy(), i0.copy()
    R = r0.copy()

    # --- output buffers ----------------------------------------------------
    t = np.arange(n_steps + 1) * h
    if record == "full":
        out_S = np.empty((n_steps + 1,) + flat, dtype=float)
        out_I = np.empty_like(out_S)
        out_R = np.empty_like(out_S)
        out_inc = np.zeros_like(out_S)
        out_S[0], out_I[0], out_R[0] = S, I, R
    else:
        out_S = out_I = out_R = out_inc = None

    min_S, min_I, min_R = S.copy(), I.copy(), R.copy()
    max_N = S + I + R
    t_min_S = np.zeros(flat, dtype=float)

    # --- time stepping -----------------------------------------------------
    for n in range(n_steps):
        t_next = (n + 1) * h

        if n_delay > 0:
            slot = (n + 1) % n_delay
            Sd, Id = buf_S[slot], buf_I[slot]
        else:
            Sd, Id = S, I

        if beta_fn is None:
            b = bet
        else:
            t_eval = t_next if beta_mode == "current" else t_next - tau
            b = np.broadcast_to(
                np.asarray(beta_fn(t_eval), dtype=float).reshape(-1), flat)

        flux = b * Sd * Id                      # infection term beta*S(t-tau)*I(t-tau)

        S_new = (S + h * Lam - h * flux) / den_S
        I_new = (I + h * flux) / den_I
        R_new = (R + h * gam * I_new) / den_R

        if n_delay > 0:
            buf_S[slot] = S_new
            buf_I[slot] = I_new
        S, I, R = S_new, I_new, R_new

        newly_min = S < min_S
        if newly_min.any():
            t_min_S = np.where(newly_min, t_next, t_min_S)
        np.minimum(min_S, S, out=min_S)
        np.minimum(min_I, I, out=min_I)
        np.minimum(min_R, R, out=min_R)
        np.maximum(max_N, S + I + R, out=max_N)

        if record == "full":
            out_S[n + 1], out_I[n + 1], out_R[n + 1] = S, I, R
            out_inc[n + 1] = flux

    # The incidence is a flux across a step, so it has no value at the initial
    # node; carry the first computed value back to index 0 rather than leaving a
    # zero there, which would otherwise show up as a spurious outlier whenever
    # the incidence is compared against data on a log scale.
    if record == "full" and n_steps >= 1:
        out_inc[0] = out_inc[1]

    def _traj(a):
        return None if a is None else a.reshape((n_steps + 1,) + sweep)

    return Solution(
        h=h, n_delay=n_delay, t=t,
        S=_traj(out_S), I=_traj(out_I), R=_traj(out_R), incidence=_traj(out_inc),
        min_S=min_S.reshape(sweep), min_I=min_I.reshape(sweep),
        min_R=min_R.reshape(sweep), max_N=max_N.reshape(sweep),
        t_min_S=t_min_S.reshape(sweep),
    )
