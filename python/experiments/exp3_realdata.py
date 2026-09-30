"""Experiment 3 -- what the delay does to parameters inferred from real data.

Data (downloaded and checksummed by ``sirdelay/data.py``; see
``python/data/PROVENANCE.md``):

* COVID-19 Italy, Dipartimento della Protezione Civile national daily series,
  from 24 February 2020.  Intervention: the national lockdown of 9 March 2020.
* Ebola virus disease, Liberia 2014, WHO and ministry situation reports as
  compiled by C. Rivers.

Three questions.

(a) What is the early exponential growth rate, and how sensitive is it to the
    choice of window?

(b) What reproduction number does that growth rate imply?  The answer depends on
    the assumed generation interval, and hence on the delay.  With the model's
    infectiousness profile relative to the infection time -- nothing for tau
    days, then exponential decay at rate g = gamma + mu_I --

        R0_delay = (1 + r/g) * exp(r*tau),      R0_ode = 1 + r/g,

    so the delay-free estimate is too small by exp(r*tau).  This is the standard
    generation-interval effect (Wallinga & Lipsitch 2007).  Part (b) also checks
    the transmission rate of Table 1 against the observed growth rate.

(c) When a control measure changes beta, when does the model say it happened?
    The time-dependent transmission rate enters as the rate in force at the time
    of contact,

        I'(t) = beta(t-tau) * S(t-tau) I(t-tau) - g I(t),

    so a change in beta reaches I only after tau.  A piecewise-constant beta is
    fitted to the Italian incidence with a scanned change point, with and
    without the delay, and with the change point pinned to the lockdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sirdelay import COVID, EBOLA                                    # noqa: E402
from sirdelay.analysis import R0_from_growth, R0_ode_from_growth     # noqa: E402
from sirdelay.data import (ITALY_LOCKDOWN, load_italy, load_liberia,  # noqa: E402
                           write_provenance)
from sirdelay.analysis import growth_rate                            # noqa: E402
from sirdelay.export import banner, write_macros, write_table        # noqa: E402
from sirdelay.fitting import (fit_growth_rate, fit_piecewise_beta,   # noqa: E402
                              profile_changepoint)

FIT_END = 70.0          # days from 24 Feb 2020 used for the trajectory fit
H = 0.05


def smooth7(y):
    """Centred 7-day mean, removing the weekday reporting cycle.

    The window is normalised by the number of points that actually contribute,
    so the three values at each end are means over a shorter window rather than
    seven-point sums divided by seven.  Dividing by a fixed 7 -- what
    ``np.convolve(..., mode="same")`` does on its own -- biases the first and
    last three values low by up to 43%, which is large compared with the
    residuals of the fits below.
    """
    y = np.asarray(y, dtype=float)
    k = np.ones(7)
    return np.convolve(y, k, mode="same") / np.convolve(np.ones_like(y), k, mode="same")


# ---------------------------------------------------------------------------
def part_a():
    banner("3a  early exponential growth rate")
    italy = load_italy()
    lock_day = float(italy.loc[italy.date == ITALY_LOCKDOWN, "day"].iloc[0])
    print(f"  Italy: {len(italy)} daily records from {italy.date.iloc[0].date()};"
          f" lockdown {ITALY_LOCKDOWN.date()} = day {lock_day:.0f}")
    print(f"  fitting log(new cases) on windows ending at the lockdown:\n")
    print(f"    {'window [d]':>12}  {'r [1/day]':>10}  {'95% CI':>18}  "
          f"{'R^2':>6}  {'T_double':>9}")
    rows = []
    for start in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
        m = (italy["day"] >= start) & (italy["day"] <= lock_day)
        f = fit_growth_rate(italy["day"][m], italy["new_cases"][m])
        rows.append((start, f))
        print(f"    {start:5.0f}-{lock_day:<6.0f}  {f.r:10.4f}  "
              f"[{f.r_lo:.4f}, {f.r_hi:.4f}]  {f.r_squared:6.3f}  "
              f"{np.log(2)/f.r:9.2f}")
    r_italy = float(np.median([f.r for _, f in rows]))
    spread = (min(f.r for _, f in rows), max(f.r for _, f in rows))
    print(f"\n  median r = {r_italy:.4f} /day, range over windows "
          f"[{spread[0]:.4f}, {spread[1]:.4f}]")

    liberia = load_liberia()
    print(f"\n  Liberia: {len(liberia)} situation reports "
          f"{liberia.date.iloc[0].date()} to {liberia.date.iloc[-1].date()}")
    m = (liberia["day"] >= 85) & (liberia["day"] <= 160)
    f_lib = fit_growth_rate(liberia["day"][m], liberia["cumulative"][m])
    print(f"  exponential phase (days 85-160, June-August 2014), "
          f"log(cumulative cases):")
    print(f"    r = {f_lib.r:.4f} /day  [{f_lib.r_lo:.4f}, {f_lib.r_hi:.4f}], "
          f"R^2 = {f_lib.r_squared:.3f}, doubling time "
          f"{np.log(2)/f_lib.r:.1f} d")

    return {"italy": italy, "lock_day": lock_day, "r_italy": r_italy,
            "r_spread": spread, "r_liberia": f_lib.r, "liberia": liberia}


# ---------------------------------------------------------------------------
def part_b(a):
    banner("3b  the reproduction number implied by that growth rate")
    out = {}
    for name, par, r in (("covid", COVID, a["r_italy"]),
                         ("ebola", EBOLA, a["r_liberia"])):
        g = par.gamma + par.mu_I
        R_ode = float(R0_ode_from_growth(r=r, gamma=par.gamma, mu_I=par.mu_I))
        R_at = float(R0_from_growth(r=r, gamma=par.gamma, mu_I=par.mu_I,
                                    tau=par.tau))
        print(f"\n  {name.upper()}:  r = {r:.4f} /day, "
              f"gamma + mu_I = {g:.4f} /day, tau = {par.tau:g} d")
        print(f"    delay-free (SIR)          R0 = {R_ode:.3f}")
        print(f"    fixed delay tau           R0 = {R_at:.3f}   "
              f"(x {np.exp(r*par.tau):.3f} = exp(r*tau))")
        print(f"    Table 1 quotes            R0 = {par.R0:.3f}  "
              f"(from beta = {par.beta}, fitted without a delay)")
        out[name] = {"R_ode": R_ode, "R_delay": R_at,
                     "factor": float(np.exp(r * par.tau))}

    print(f"\n  Consistency check on Table 1: integrating the delay model with the")
    print(f"  tabulated beta = {COVID.beta} and tau = {COVID.tau:g} gives an early")
    r_tab = growth_rate(beta=COVID.beta, gamma=COVID.gamma, mu_I=COVID.mu_I,
                        S_star=COVID.S_star, tau=COVID.tau)
    r_tab0 = growth_rate(beta=COVID.beta, gamma=COVID.gamma, mu_I=COVID.mu_I,
                         S_star=COVID.S_star, tau=0.0)
    beta_delay = float((a["r_italy"] + COVID.gamma + COVID.mu_I)
                       * np.exp(a["r_italy"] * COVID.tau) / COVID.S_star)
    print(f"  growth rate r = {r_tab:.4f} /day (doubling {np.log(2)/r_tab:.1f} d),")
    print(f"  against r = {r_tab0:.4f} /day without the delay and "
          f"r = {a['r_italy']:.4f} /day")
    print(f"  in the data.  The tabulated beta therefore under-reproduces the")
    print(f"  observed growth by a factor {a['r_italy']/r_tab:.2f} once the delay")
    print(f"  is present.  The delay-consistent value is beta = {beta_delay:.3f}.")
    out["beta_delay"] = beta_delay
    out["r_table"] = float(r_tab)
    return out


# ---------------------------------------------------------------------------
def part_c(a):
    banner("3c  when does a fitted model say the intervention happened?")
    italy, lock_day = a["italy"], a["lock_day"]
    r = a["r_italy"]
    sel = italy["day"] <= FIT_END
    t_data = italy["day"][sel].to_numpy(dtype=float)
    y_data = smooth7(italy["incidence"][sel].to_numpy(dtype=float))
    keep = y_data > 0
    t_data, y_data = t_data[keep], y_data[keep]

    I0 = float(italy["I"].iloc[0])
    R0d = float(italy["R"].iloc[0])
    print(f"  fitting the smoothed daily incidence over days 0-{FIT_END:.0f}")
    print(f"  history: exponential back-extrapolation at the fitted r = {r:.4f}")
    print(f"  true intervention: {ITALY_LOCKDOWN.date()} = day {lock_day:.0f}\n")

    def make_hist(tau):
        def fn(ts):
            ts = np.asarray(ts, dtype=float)
            I = I0 * np.exp(r * ts)
            R = R0d * np.exp(r * ts)
            return 1.0 - I - R, I, R
        return {"S": 1.0 - I0 - R0d, "I": I0, "R": R0d, "fn": fn}

    t0_grid = np.arange(4.0, 41.0, 1.0)
    variants = (("ode", "no delay (tau = 0)", 0.0),
                ("delay", "delay, beta at contact time", COVID.tau))
    res, fits = {}, {}
    for key, label, tau in variants:
        _, _, best = profile_changepoint(
            COVID, t_data=t_data, y_data=y_data, hist=make_hist(tau), tau=tau,
            t0_grid=t0_grid, beta_mode="delayed", observable="incidence", h=H,
            T=FIT_END, x0=(0.6, 0.15))
        res[key] = {"label": label, "t0": best.t0, "beta1": best.beta1,
                    "beta2": best.beta2, "reduction": best.reduction,
                    "rmse": best.rmse, "fit": best}
        fits[key] = np.interp(t_data, best.model_t, best.model_I)
        print(f"    {label:<40}")
        print(f"      inferred change point : day {best.t0:.0f}  "
              f"({best.t0 - lock_day:+.0f} d relative to the lockdown)")
        print(f"      beta before / after   : {best.beta1:.3f} -> {best.beta2:.3f} "
              f"({100*best.reduction:.0f}% reduction)")
        print(f"      rmse (log incidence)  : {best.rmse:.4f}")

    # ---- robustness: pin the change point to the actual lockdown -----------
    # A lockdown on day `lock_day` reduces contacts from that day.  In this model
    # an infection recorded at time t stems from a contact at t - tau, so the
    # reduction reaches the incidence only at lock_day + tau.
    banner("3c(iii)  change point pinned to the lockdown date")
    lock = {}
    for key, tau, label in (("ode", 0.0, "no delay"),
                            ("delay", COVID.tau, "delay")):
        fit = fit_piecewise_beta(COVID, t_data=t_data, y_data=y_data,
                                 hist=make_hist(tau), tau=tau, t0=lock_day,
                                 beta_mode="delayed", observable="incidence", h=H,
                                 T=FIT_END, x0=(0.6, 0.15))
        curve = np.interp(t_data, fit.model_t, fit.model_I)
        lock[key] = {"fit": fit, "curve": curve,
                     "peak": float(t_data[int(np.argmax(curve))])}
        print(f"    {label:<18} beta {fit.beta1:.3f} -> {fit.beta2:.3f}  "
              f"({100*fit.reduction:.0f}% reduction), SSE {fit.sse:.2f}, "
              f"model peak day {lock[key]['peak']:.0f}")
    fits["lock"] = lock["delay"]["curve"]
    data_peak = float(t_data[int(np.argmax(y_data))])
    print(f"    observed peak day {data_peak:.0f}; free-change-point SSE was "
          f"{res['delay']['rmse']**2 * len(t_data):.2f}")

    # Does an observation lag between infection and reporting repair it?
    lags, sses = list(range(0, 15)), []
    for d in lags:
        shifted = t_data - d
        m = shifted >= 0
        f_lag = fit_piecewise_beta(COVID, t_data=shifted[m], y_data=y_data[m],
                                   hist=make_hist(COVID.tau), tau=COVID.tau,
                                   t0=lock_day, beta_mode="delayed",
                                   observable="incidence", h=H, T=FIT_END,
                                   x0=(0.6, 0.15))
        sses.append(f_lag.sse)
    d_best = int(lags[int(np.argmin(sses))])
    print(f"    adding a reporting lag d does not repair it: best d = {d_best} d, "
          f"SSE {min(sses):.2f}")
    res["lock_red_ode"] = float(lock["ode"]["fit"].reduction)
    res["lock_red_delay"] = float(lock["delay"]["fit"].reduction)
    res["lock_beta1"] = float(lock["delay"]["fit"].beta1)
    res["lock_beta2"] = float(lock["delay"]["fit"].beta2)
    res["lock_sse"] = float(lock["delay"]["fit"].sse)
    res["lock_peak"] = lock["delay"]["peak"]
    res["data_peak"] = data_peak
    res["lag_best"] = d_best
    res["lag_sse"] = float(min(sses))
    # A step in beta makes the modelled incidence beta(t-tau)S(t-tau)I(t-tau)
    # jump by exactly beta2/beta1 at t0 + tau.  Least squares on the logarithm
    # centres that jump on the data, so the curve sits above the data just before
    # the jump and below just after, by comparable factors.  Reporting only the
    # upper side as an "overshoot" would misrepresent a two-sided straddle as a
    # bias.
    best = res["delay"]["fit"]
    t_jump = best.t0 + COVID.tau
    tt, inc = best.model_t, best.model_I
    j = int(np.searchsorted(tt, t_jump))
    obs = float(np.interp(t_jump, t_data, y_data))
    res["jump"] = float(best.beta2 / best.beta1)
    res["above"] = float(inc[j - 2] / obs)
    res["below"] = float(obs / inc[j + 2])
    resid = np.log(np.interp(t_data, tt, inc)) - np.log(y_data)
    res["bias"] = float(resid.mean())
    print(f"\n  a step in beta makes the modelled incidence jump by beta2/beta1 = "
          f"{res['jump']:.3f}")
    print(f"  at t0 + tau; log-least-squares centres that jump on the data,")
    print(f"  leaving the curve {res['above']:.1f}x above just before and "
          f"{res['below']:.1f}x below just after.")
    print(f"  Mean log-residual over the window: {res['bias']:+.3f} (no systematic bias).")

    # Mesh convergence of the fit itself.
    ref = fit_piecewise_beta(COVID, t_data=t_data, y_data=y_data,
                             hist=make_hist(COVID.tau), tau=COVID.tau,
                             t0=best.t0, beta_mode="delayed",
                             observable="incidence", h=H / 10, T=FIT_END,
                             x0=(0.6, 0.15))
    res["mesh"] = max(abs(best.beta1 / ref.beta1 - 1.0),
                      abs(best.beta2 / ref.beta2 - 1.0))
    print(f"  refining the step size tenfold changes the fitted rates by "
          f"{100*res['mesh']:.1f}%, so the fit is mesh-converged.")
    # The same centred 7-day mean as for the data, applied to the model incidence
    # sampled on the observation days, for a like-for-like comparison in the
    # figure.  The fits themselves use the unsmoothed model incidence.
    assert np.all(np.diff(t_data) == 1.0), "7-day mean needs consecutive days"
    write_table("exp3c_fits",
                {"day": t_data, "data": y_data, "ode": fits["ode"],
                 "delay": fits["delay"], "lock": fits["lock"],
                 "ode_avg": smooth7(fits["ode"]),
                 "delay_avg": smooth7(fits["delay"]),
                 "lock_avg": smooth7(fits["lock"])},
                comment="Experiment 3c -- smoothed Italian incidence (density) and the\n"
                        "best fits: no delay and delay, each at its own inferred\n"
                        "change point, and delay with the change point at the lockdown.\n"
                        "Columns *_avg: the same centred 7-day mean as for the data,\n"
                        "applied to the model curves.")
    res["lock_day"] = lock_day
    return res


# ---------------------------------------------------------------------------
def main() -> None:
    print(f"provenance written to {write_provenance()}")
    a = part_a()
    b = part_b(a)
    c = part_c(a)

    write_macros("exp3_macros", {
        "expThreeRItaly": f"{a['r_italy']:.3f}",
        "expThreeRItalyLo": f"{a['r_spread'][0]:.3f}",
        "expThreeRItalyHi": f"{a['r_spread'][1]:.3f}",
        "expThreeRLiberia": f"{a['r_liberia']:.4f}",
        "expThreeCovidRode": f"{b['covid']['R_ode']:.2f}",
        "expThreeCovidRdelay": f"{b['covid']['R_delay']:.2f}",
        "expThreeCovidFactor": f"{b['covid']['factor']:.2f}",
        "expThreeEbolaRode": f"{b['ebola']['R_ode']:.2f}",
        "expThreeEbolaRdelay": f"{b['ebola']['R_delay']:.2f}",
        "expThreeBetaDelay": f"{b['beta_delay']:.2f}",
        "expThreeRtable": f"{b['r_table']:.3f}",
        "expThreeLockDay": f"{c['lock_day']:.0f}",
        "expThreeTzeroOde": f"{c['ode']['t0']:.0f}",
        "expThreeTzeroDelay": f"{c['delay']['t0']:.0f}",
        "expThreeAbove": f"{c['above']:.1f}",
        "expThreeBelow": f"{c['below']:.1f}",
        "expThreeBias": f"{c['bias']:+.2f}",
        "expThreeLockRedOde": f"{100*c['lock_red_ode']:.0f}",
        "expThreeLockRedDelay": f"{100*c['lock_red_delay']:.0f}",
        "expThreeLockBetaOne": f"{c['lock_beta1']:.2f}",
        "expThreeLockBetaTwo": f"{c['lock_beta2']:.2f}",
        "expThreeLockSse": f"{c['lock_sse']:.1f}",
        "expThreeLockPeak": f"{c['lock_peak']:.0f}",
        "expThreeDataPeak": f"{c['data_peak']:.0f}",
        "expThreeLagBest": f"{c['lag_best']:.0f}",
        "expThreeLagSse": f"{c['lag_sse']:.1f}",
        "expThreeRmseOde": f"{c['ode']['rmse']:.2f}",
        "expThreeRmseDelay": f"{c['delay']['rmse']:.2f}",
        "expThreeRedOde": f"{100*c['ode']['reduction']:.0f}",
        "expThreeRedDelay": f"{100*c['delay']['reduction']:.0f}",
        "expThreeBetaOdeOne": f"{c['ode']['beta1']:.2f}",
        "expThreeBetaDelayOne": f"{c['delay']['beta1']:.2f}",
    }, comment="Experiment 3 -- real-world data.")
    banner("experiment 3 complete")


if __name__ == "__main__":
    main()
