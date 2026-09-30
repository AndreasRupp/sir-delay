"""Experiment 1 -- how sharp are the positivity conditions of Theorem 2?

Theorem 2 asks for

    (A)  N0_bar < 2 sqrt(Lambda/beta)      (B)  Lambda*beta <= 4 mu^2

and concludes positivity *for every* tau > 0.  Being uniform in tau and in the
history, the conditions cannot be sharp for any individual configuration; the
meaningful questions are whether they are attained in the worst case, and how
much slack is left.

Condition (A) rests on the arithmetic-geometric mean bound

    S*I <= (S+I)^2/4 <= N^2/4,

attained when S = I = N/2.  That the dynamics really realize this worst case is
what part (b) checks.  There is an analytic prediction: on the first delay
interval the history is frozen, so S obeys the linear equation
S' = Lambda - beta*S0*I0 - mu_S*S, which decays towards
(Lambda - beta*S0*I0)/mu_S.  That level is negative exactly when beta*S0*I0 >
Lambda, and for large tau there is enough time to reach zero.  Hence the exact
threshold as tau -> infinity is

    beta * S0 * I0 <= Lambda,

which at the worst-case split S0 = I0 = N0_bar/2 is precisely (A).  So (A) is
predicted to be attained as tau -> infinity, i.e. sharp.

Condition (B) is not redundant: part (e) shows that above a critical beta
positivity is lost for histories of any size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sirdelay import solve                                        # noqa: E402
from sirdelay.export import (banner, sci, write_macros,           # noqa: E402
                             write_matrix, write_table)

LAMBDA, GAMMA, MU = 1.0, 0.3, 0.9          # the dimensionless set of Section 3
WORST = (0.5, 0.5, 0.0)                    # AM-GM worst-case history split
REALISTIC = (0.90, 0.05, 0.05)

TAUS = (0.1, 1.0, 10.0)

# Largest beta admitted by the balance condition (B) of Theorem 2.
BETA_MAX_B = 4.0 * MU**2 / LAMBDA          # = 3.24 for this parameter set


def _horizon(tau: float) -> tuple[float, float]:
    """Final time and target step size for a given delay."""
    T = max(80.0, 15.0 * tau)
    h = 0.05 if tau <= 20.0 else 0.2
    return T, h


def stays_positive(beta, N0_bar, *, tau, split=WORST) -> np.ndarray:
    """Vectorised test: does the discrete solution keep S, I, R >= 0?

    Runs that lose positivity subsequently diverge, so overflow to inf/nan is
    expected and harmless -- a nan compares false against ``>= 0`` and the run is
    correctly classified as non-positive.  The warnings are silenced rather than
    guarded against, since the verdict is recorded by the running minimum long
    before any overflow occurs.
    """
    beta = np.asarray(beta, dtype=float)
    N0_bar = np.asarray(N0_bar, dtype=float)
    fS, fI, fR = split
    T, h = _horizon(tau)
    with np.errstate(over="ignore", invalid="ignore"):
        sol = solve(Lambda=LAMBDA, beta=beta, gamma=GAMMA,
                    mu_S=MU, mu_I=MU, mu_R=MU, tau=tau,
                    S0=fS * N0_bar, I0=fI * N0_bar, R0=fR * N0_bar,
                    T=T, h=h, record="summary")
        return sol.positive


def critical_N0(beta, *, tau, split=WORST, lo=1e-3, hi=60.0, iters=40):
    """Largest history level N0_bar for which positivity still holds, per beta.

    Bisection assumes monotonicity in N0_bar; :func:`check_monotonicity` verifies
    that assumption on a raster before the boundaries are trusted.
    """
    beta = np.asarray(beta, dtype=float)
    lo = np.full(beta.shape, lo)
    hi = np.full(beta.shape, hi)
    # Guard: if even the smallest level fails, report NaN rather than a bogus root.
    ok_lo = stays_positive(beta, lo, tau=tau, split=split)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = stays_positive(beta, mid, tau=tau, split=split)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    out = 0.5 * (lo + hi)
    return np.where(ok_lo, out, np.nan)


# ---------------------------------------------------------------------------
def _non_monotone_columns(ok):
    """Indices of beta columns whose positive set is *not* a lower interval."""
    bad = []
    for j in range(ok.shape[1]):
        col = ok[:, j]
        if col.any():
            last = int(np.max(np.where(col)[0]))
            if not col[: last + 1].all():
                bad.append(j)
    return bad


def check_monotonicity() -> dict:
    """Raster check that the positive set is an interval in N0_bar.

    Monotonicity is what justifies locating the boundary by bisection.  It holds
    wherever the balance condition (B) does, and *fails* above it -- which is the
    subject of :func:`part_e`.
    """
    banner("1b(i)  monotonicity check for the bisection")
    betas = np.geomspace(0.05, 8.0, 80)
    levels = np.linspace(0.02, 6.0, 140)
    B, L = np.meshgrid(betas, levels)
    ok = stays_positive(B.ravel(), L.ravel(), tau=1.0).reshape(B.shape)
    bad = _non_monotone_columns(ok)
    below = [j for j in bad if betas[j] <= BETA_MAX_B]
    above = [j for j in bad if betas[j] > BETA_MAX_B]
    print(f"  balance condition (B) holds for beta <= 4 mu^2/Lambda = {BETA_MAX_B:.2f}")
    print(f"  non-monotone beta columns below that threshold : {len(below)} "
          f"(of {int((betas <= BETA_MAX_B).sum())})")
    print(f"  non-monotone beta columns above that threshold : {len(above)} "
          f"(of {int((betas > BETA_MAX_B).sum())})")
    if above:
        print(f"    smallest such beta: {betas[min(above)]:.3f}")
    write_matrix("exp1b_raster", betas, levels, ok.astype(float),
                 xname="beta", yname="N0bar", zname="positive",
                 comment="Experiment 1b -- raster of numerically observed positivity,\n"
                         "tau = 1, worst-case history split S0 = I0 = N0_bar/2.\n"
                         "Lambda = 1, mu = 0.9, gamma = 0.3.")
    return {"monotone_below": not below, "n_above": len(above),
            "beta_first": float(betas[min(above)]) if above else float("nan")}


def part_b() -> dict:
    """True positivity boundary against the threshold of Theorem 2.

    Restricted to beta <= 4 mu^2 / Lambda, i.e. to the region where the balance
    condition (B) holds and the positive set is monotone in N0_bar.  The column
    ``naive`` is the weaker threshold sqrt(Lambda/beta) that the cruder estimate
    S*I <= N^2 would give; it is retained only for comparison.
    """
    banner("1b(ii)  true boundary N0_bar*(beta) against the threshold of Theorem 2")
    betas = np.geomspace(0.05, BETA_MAX_B, 120)
    cols = {"beta": betas,
            "naive": np.sqrt(LAMBDA / betas),
            "thm": 2.0 * np.sqrt(LAMBDA / betas)}
    summary = {}
    for tau in TAUS:
        crit = critical_N0(betas, tau=tau)
        key = f"{tau:g}".replace(".", "p")
        cols[f"crit{key}"] = crit
        ratio = crit / (2.0 * np.sqrt(LAMBDA / betas))
        summary[tau] = (float(np.nanmin(ratio)), float(np.nanmedian(ratio)),
                        float(np.nanmax(ratio)))
        print(f"    tau = {tau:6g}:  N0_bar*(beta) / [2 sqrt(Lambda/beta)]")
        print(f"                  min {summary[tau][0]:.3f}   median "
              f"{summary[tau][1]:.3f}   max {summary[tau][2]:.3f}")
    write_table("exp1b_boundary", cols,
                comment="Experiment 1b -- numerically determined critical history level\n"
                        "N0_bar*(beta) (bisection, worst-case split S0=I0=N0_bar/2)\n"
                        "against the threshold 2 sqrt(Lambda/beta) of Theorem 2 ('thm')\n"
                        "and the weaker sqrt(Lambda/beta) of the cruder estimate\n"
                        "('naive').  Lambda = 1, mu = 0.9, gamma = 0.3.")
    return summary


def part_c() -> dict:
    """How much of the slack is due to uniformity over the history split.

    The sweep is restricted to beta <= 4 mu^2 / Lambda, i.e. to the region where
    the balance condition (B) holds.  Beyond it positivity is lost for histories
    of any size (see :func:`part_e`), so the critical level is not governed by the
    first-interval balance beta*S0*I0 = Lambda and comparing against that
    prediction would be meaningless.
    """
    banner("1c  dependence on the history split")
    betas = np.geomspace(0.05, BETA_MAX_B, 60)
    out = {"beta": betas}
    res = {}
    for label, split in (("worst", WORST), ("realistic", REALISTIC)):
        crit = critical_N0(betas, tau=100.0, split=split)
        out[label] = crit
        # analytic tau -> infinity prediction: beta * fS * fI * N0^2 = Lambda
        pred = np.sqrt(LAMBDA / (betas * split[0] * split[1]))
        out[f"pred_{label}"] = pred
        dev = np.abs(crit - pred) / pred
        res[label] = float(np.nanmedian(dev))
        res[f"{label}_max"] = float(np.nanmax(dev))
        print(f"    {label:10s} split {split}:  relative deviation from the analytic "
              f"prediction: median {res[label]:.3e}, max {res[label + '_max']:.3e}")
    write_table("exp1c_split", out,
                comment="Experiment 1c -- critical history level at tau = 100 for two\n"
                        "history splits, against the analytic large-tau prediction\n"
                        "beta * S0 * I0 = Lambda.  Restricted to beta <= 4 mu^2 / Lambda.")
    return res


def part_d() -> dict:
    """Critical delay at a parameter point inside the gap."""
    banner("1d  the true threshold is tau-dependent although the condition is not")
    beta = 0.8
    naive = np.sqrt(LAMBDA / beta)
    thm = 2.0 * naive
    taus = np.geomspace(0.05, 300.0, 60)
    crit = np.array([float(critical_N0(np.array([beta]), tau=float(t))[0])
                     for t in taus])
    print(f"  beta = {beta}: threshold of Theorem 2 {thm:.4f} "
          f"(cruder estimate would give {naive:.4f})")
    for t, c in zip(taus[::12], crit[::12]):
        print(f"    tau = {t:8.3f}   N0_bar* = {c:7.4f}   "
              f"= {c/thm:5.2f} x the threshold of Theorem 2")
    print(f"    tau -> large  N0_bar* = {crit[-1]:7.4f}   "
          f"= {crit[-1]/thm:5.3f} x the threshold of Theorem 2")
    write_table("exp1d_tau", {"tau": taus, "crit": crit,
                              "naive": np.full_like(taus, naive),
                              "thm": np.full_like(taus, thm)},
                comment="Experiment 1d -- critical history level as a function of the\n"
                        "delay, at beta = 0.8, Lambda = 1, mu = 0.9, worst-case split.\n"
                        "The sufficient condition is uniform in tau (horizontal line\n"
                        "'thm'); the true threshold is not.")
    return {"limit_ratio": float(crit[-1] / thm), "small_tau": float(crit[0] / thm)}


def part_e() -> dict:
    """The balance condition (B) is not redundant: small histories are not safe.

    Condition (A) alone bounds only the *initial* population.  If (B) fails, the
    population climbs to its asymptotic level Lambda/mu, and the epidemic that
    then takes off drives the product S*I above Lambda/beta no matter how small
    the history was.  Numerically this shows up as a loss of monotonicity: for
    large beta the *smallest* histories fail while intermediate ones survive,
    because a small initial infection lets S recover to S* before the outbreak,
    producing a much larger overshoot.
    """
    banner("1e  condition (B) is not redundant -- arbitrarily small histories fail")
    print(f"  balance condition (B) of Theorem 2:  beta <= 4 mu^2 / Lambda = "
          f"{BETA_MAX_B:.4f}")
    print(f"  asymptotic population level Lambda/mu = {LAMBDA/MU:.4f}, so the "
          f"largest\n  product S*I the model can settle at is "
          f"(Lambda/mu)^2/4 = {(LAMBDA/MU)**2/4:.4f}, and\n  positivity requires "
          f"beta * that <= Lambda, i.e. exactly beta <= {BETA_MAX_B:.4f}.\n")

    tiny = 1e-4                       # a history four orders below the threshold
    betas = np.geomspace(0.5 * BETA_MAX_B, 4.0 * BETA_MAX_B, 200)
    ok = stays_positive(betas, np.full_like(betas, tiny), tau=1.0,
                        split=REALISTIC)
    first = int(np.argmin(ok)) if not ok.all() else -1
    beta_crit = float(betas[first]) if first >= 0 else float("nan")
    print(f"  history N0_bar = {tiny:g} (split {REALISTIC}), tau = 1:")
    print(f"    smallest beta at which positivity is lost : {beta_crit:.4f}")
    print(f"    ratio to the (B) threshold                : "
          f"{beta_crit / BETA_MAX_B:.3f}")
    print(f"    -> for beta above this, no history however small keeps S >= 0,")
    print(f"       so condition (A) alone cannot imply positivity.")

    # How the failure threshold depends on the history size.
    levels = np.geomspace(1e-6, 1.0, 40)
    crit_beta = []
    for lv in levels:
        okk = stays_positive(betas, np.full_like(betas, lv), tau=1.0,
                             split=REALISTIC)
        crit_beta.append(float(betas[int(np.argmin(okk))]) if not okk.all()
                         else float("nan"))
    write_table("exp1e_balance",
                {"N0bar": levels, "betacrit": crit_beta,
                 "naive": np.full_like(levels, MU**2 / LAMBDA),
                 "thm": np.full_like(levels, BETA_MAX_B)},
                comment="Experiment 1e -- critical transmission rate above which\n"
                        "positivity is lost, as a function of the history size, at\n"
                        "tau = 1 and the realistic split (0.90,0.05,0.05).  It is\n"
                        "essentially independent of the history, which is what the\n"
                        "balance condition Lambda*beta <= 4 mu^2 guards against.")
    return {"beta_crit": beta_crit, "ratio": beta_crit / BETA_MAX_B}


# ---------------------------------------------------------------------------
def main() -> None:
    mono = check_monotonicity()
    boundary = part_b()
    split = part_c()
    part_d()
    balance = part_e()

    write_macros("exp1_macros", {
        "expOneSmallTauRatioMax": f"{boundary[min(TAUS)][2]:.1f}",
        "expOneSplitErrReal": f"{100*split['realistic']:.1f}",
        "expOneSplitMaxReal": f"{100*split['realistic_max']:.0f}",
        "expOneBetaCrit": f"{balance['beta_crit']:.3f}",
    }, comment="Experiment 1 -- sharpness of the positivity conditions.")
    banner("summary of experiment 1")
    print(f"  bisection justified (positive set monotone in N0_bar below (B)): "
          f"{mono['monotone_below']}")
    print(f"  (A) the history condition 2 sqrt(Lambda/beta) is attained: at "
          f"tau = {max(TAUS):g}")
    print(f"      the measured threshold is {boundary[max(TAUS)][0]:.3f} x it, and "
          f"the agreement with")
    print(f"      the analytic large-tau prediction beta*S0*I0 = Lambda deviates "
          f"by {split['worst']:.1e} (relative).")
    print(f"  (B) the balance condition is not redundant: above beta = "
          f"{balance['beta_crit']:.3f}")
    print(f"      ({balance['ratio']:.3f} x the threshold {BETA_MAX_B:.2f}) "
          f"positivity fails for")
    print(f"      histories of any size.")
    banner("experiment 1 complete")

if __name__ == "__main__":
    main()
