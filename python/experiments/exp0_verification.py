"""Experiment 0 -- verification of the scheme and of discrete positivity.

Two checks (Section 4.1 of the manuscript):

1. *Convergence order.*  The scheme treats the linear part implicitly and the
   delayed nonlinearity explicitly from stored grid values, so it is a
   first-order method.  The method-of-steps regularity gain of Theorem 1 means
   the solution is smooth away from t = 0, so no order reduction is expected
   despite the derivative jumps at multiples of tau.

2. *Discrete positivity, unconditionally in h.*  Summing the three discrete
   equations gives

       N^{n+1} (1 + h*mu) <= N^n + h*Lambda,   hence   N^{n+1} <= max{N^n, Lambda/mu},

   and under the hypotheses of Theorem 2, via S*I <= N**2/4, the S-numerator
   satisfies S^n + h*Lambda - h*beta*S^{n+1-m}*I^{n+1-m} >= S^n >= 0.  Both statements are
   free of any step-size restriction, so the scheme should inherit Theorem 2 for
   *every* h -- including the extreme case h = tau (a single step per delay
   interval).  This check sweeps h over eleven octaves to look for a
   counterexample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sirdelay import ACADEMIC, COVID, solve                       # noqa: E402
from sirdelay.export import (banner, sci, write_macros,          # noqa: E402
                             write_table)


def run(par, *, tau, S0, I0, R0, T, h, record="full"):
    return solve(Lambda=par.Lambda, beta=par.beta, gamma=par.gamma,
                 mu_S=par.mu_S, mu_I=par.mu_I, mu_R=par.mu_R, tau=tau,
                 S0=S0, I0=I0, R0=R0, T=T, h=h, record=record)


# ---------------------------------------------------------------------------
def convergence(par, *, tau, T, S0, I0, R0, k_ref=14, ks=range(1, 11)):
    ref = run(par, tau=tau, S0=S0, I0=I0, R0=R0, T=T, h=tau / 2**k_ref)
    exact = np.array([ref.S[-1], ref.I[-1], ref.R[-1]])
    rows = []
    for k in ks:
        sol = run(par, tau=tau, S0=S0, I0=I0, R0=R0, T=T, h=tau / 2**k)
        approx = np.array([sol.S[-1], sol.I[-1], sol.R[-1]])
        rows.append((sol.h, 2**k, float(np.max(np.abs(approx - exact)))))
    h = np.array([r[0] for r in rows])
    err = np.array([r[2] for r in rows])
    rate = np.full_like(err, np.nan)
    rate[1:] = np.log(err[:-1] / err[1:]) / np.log(2.0)
    return h, np.array([r[1] for r in rows], dtype=float), err, rate


def check_convergence() -> dict:
    banner("0.1  convergence order at t = T")
    results = {}
    for label, par, tau, T, hist in (
        ("academic", ACADEMIC, 1.0, 10.0, (0.85, 0.10, 0.10)),
        ("covid", COVID, 5.0, 60.0, (0.99, 1e-3, 0.0)),
    ):
        h, n, err, rate = convergence(par, tau=tau, T=T, S0=hist[0], I0=hist[1],
                                      R0=hist[2])
        print(f"\n  {label} (tau = {tau}, T = {T})")
        print(f"    {'N':>6}  {'h':>12}  {'error':>12}  {'rate':>6}")
        for ni, hi, ei, ri in zip(n, h, err, rate):
            rs = "  --  " if np.isnan(ri) else f"{ri:6.3f}"
            print(f"    {int(ni):6d}  {hi:12.3e}  {ei:12.4e}  {rs}")
        write_table(f"exp0_convergence_{label}",
                    {"N": n, "h": h, "error": err},
                    comment=f"Experiment 0.1 -- implicit Euler convergence, {label} set,\n"
                            f"tau = {tau}, T = {T}, reference h = tau/2^14.\n"
                            "Max-norm error in (S,I,R) at t = T.")
        results[label] = float(np.nanmean(rate[-4:]))
        print(f"    mean observed rate over the last four refinements: "
              f"{results[label]:.3f}")
    return results


# ---------------------------------------------------------------------------
def check_discrete_positivity() -> dict:
    banner("0.2  discrete positivity and population bound, unconditionally in h")
    par = ACADEMIC
    # Admissible history of Section 4.2: sup N over [-tau,0] equals 1.05.
    S0, I0, R0 = 0.85, 0.10, 0.10
    N0_bar = S0 + I0 + R0
    bound = max(N0_bar, par.Lambda / par.mu)
    cond = par.theorem2(N0_bar)
    print(f"  Lambda = {par.Lambda}, beta = {par.beta}, gamma = {par.gamma}, "
          f"mu = {par.mu}")
    print(f"  N0_bar = {N0_bar:.4f} < 2 sqrt(Lambda/beta) = "
          f"{2*par.threshold:.4f} : {cond['cond_history']}")
    print(f"  Lambda*beta = {par.Lambda * par.beta:.4f} <= 4 mu^2 = "
          f"{4*par.mu**2:.4f} : {cond['cond_balance']}")
    print(f"  predicted population bound max(N0_bar, Lambda/mu) = {bound:.6f}\n")
    print(f"    {'N':>6}  {'h':>10}  {'min(S,I,R)':>14}  {'max N':>12}  "
          f"{'<= bound':>9}")
    rows = []
    for k in range(0, 11):
        n_delay = 2**k
        h = par.tau / n_delay
        for tau in (0.5, 1.0, 5.0, 20.0):
            sol = run(par, tau=tau, S0=S0, I0=I0, R0=R0, T=200.0,
                      h=tau / n_delay, record="summary")
            lo = float(min(sol.min_S, sol.min_I, sol.min_R))
            hi = float(sol.max_N)
            rows.append((tau, n_delay, sol.h, lo, hi))
            if tau == 1.0:
                print(f"    {n_delay:6d}  {sol.h:10.4f}  {lo:14.6e}  {hi:12.8f}  "
                      f"{str(hi <= bound + 1e-12):>9}")
    lo_all = min(r[3] for r in rows)
    hi_all = max(r[4] for r in rows)
    print(f"\n  over all {len(rows)} runs (tau in 0.5,1,5,20; h from tau to tau/1024):")
    print(f"    min over all components and steps : {lo_all:.6e}")
    print(f"    max total population              : {hi_all:.10f}  "
          f"(bound {bound:.10f})")
    ok = lo_all >= 0.0 and hi_all <= bound + 1e-12
    print(f"    positivity and bound preserved for every h tested: {ok}")
    assert ok, "discrete positivity or the population bound was violated"

    write_table("exp0_positivity_h",
                {"tau": [r[0] for r in rows], "Ndelay": [r[1] for r in rows],
                 "h": [r[2] for r in rows], "minSIR": [r[3] for r in rows],
                 "maxN": [r[4] for r in rows]},
                comment="Experiment 0.2 -- discrete positivity of the implicit Euler\n"
                        "scheme under the hypotheses of Theorem 2, for step sizes\n"
                        "from h = tau down to h = tau/1024 and four delays.")
    return {"min": lo_all, "max_N": hi_all, "bound": bound, "n_runs": len(rows)}


# ---------------------------------------------------------------------------
def main() -> None:
    rates = check_convergence()
    pos = check_discrete_positivity()

    write_macros("exp0_macros", {
        "expZeroRateAcademic": f"{rates['academic']:.2f}",
        "expZeroRateCovid": f"{rates['covid']:.2f}",
        "expZeroRuns": f"{pos['n_runs']}",
        "expZeroMinSIR": sci(pos["min"]),
        "expZeroMaxN": f"{pos['max_N']:.4f}",
        "expZeroBound": f"{pos['bound']:.4f}",
    }, comment="Experiment 0 -- verification.")
    banner("experiment 0 complete")


if __name__ == "__main__":
    main()
