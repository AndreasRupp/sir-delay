# Positivity of an SIR model with delay: code and data

This repository contains the code and data for the numerical study (Section 4) of

> R. Altmann and A. Rupp, *Positivity of an SIR model with delay*.

It covers the delayed SIR model

```
S'(t) = Lambda - beta S(t-tau) I(t-tau) - mu_S S(t)
I'(t) = beta S(t-tau) I(t-tau) - (gamma + mu_I) I(t)
R'(t) = gamma I(t) - mu_R R(t)
```

discretised with the implicit Euler scheme of Section 2.3. Every figure in the
paper and every number quoted in Section 4 is computed by the scripts here.

## Contents

| Path | Contents |
| --- | --- |
| `python/sirdelay/` | Python package: solver (`model.py`), parameter sets of Table 1 (`parameters.py`), characteristic equation and diagnostics (`analysis.py`), growth-rate and `beta(t)` fitting (`fitting.py`), data download/checksums (`data.py`), pgfplots export (`export.py`) |
| `python/experiments/` | One script per experiment (`exp0` … `exp4`) and `run_all.py` |
| `python/data/` | The two public data sets as used in the paper, plus `PROVENANCE.md` (sources, licences, SHA-256 checksums) |
| `python/requirements.txt` | Minimal dependency ranges |
| `python/requirements-lock.txt` | Exact package versions used for the paper |
| `pics/fig_*.tex` | pgfplots sources of Figures 1–5 (the same files the manuscript includes) |
| `pics/sirstyle.tex` | Shared pgfplots styles |
| `pics/data/*.dat` | Generated tables that the figures plot |
| `pics/data/exp*_macros.tex` | Generated `\newcommand`s for the numbers quoted in the text |
| `figures.tex` | Standalone LaTeX document that builds all five figures |

The generated files in `pics/data/` are committed so the figures can be built
without running Python. Running the experiments overwrites them.

## Requirements

* Python 3.9 or newer, with `numpy`, `scipy` and `pandas`.
* A TeX distribution with `pgfplots` (≥ 1.18) to build the figures, e.g.
  TeX Live or MacTeX.

## Setup

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/pip install -r python/requirements-lock.txt   # exact versions used for the paper
# or: .venv/bin/pip install -r python/requirements.txt  # any recent versions
```

## Reproducing everything

```sh
cd python
../.venv/bin/python -m experiments.run_all      # about 2.5 minutes on a laptop
cd ..
latexmk -pdf figures.tex                          # or: pdflatex figures.tex
```

`run_all` runs experiments 0–4 in order and rewrites all of `pics/data/`.
`figures.tex` then typesets Figures 1–5 from those tables into `figures.pdf`.
With the pinned versions in `requirements-lock.txt` the regenerated tables
match the committed ones byte for byte.

On macOS, NumPy 2.0 with the Accelerate backend may print
`RuntimeWarning: divide by zero / overflow / invalid value encountered in matmul`
from `sirdelay/fitting.py`. These warnings are spurious and do not affect the
results.

## How each figure is generated

Every experiment can also be run on its own from inside `python/`, e.g.
`../.venv/bin/python -m experiments.exp1_sharpness`. Figure 1 uses output from
two experiments.

| Figure | Section | Figure source | Data tables | Command (run in `python/`) |
| --- | --- | --- | --- | --- |
| 1, left: convergence in the step size `h` | 4.1 | `pics/fig_verification.tex` | `exp0_convergence_academic.dat`, `exp0_convergence_covid.dat` | `python -m experiments.exp0_verification` |
| 1, right: convergence as `tau -> 0` | 4.1 | `pics/fig_verification.tex` | `exp4a_taulimit.dat` | `python -m experiments.exp4_limit_regularity` |
| 2: sharpness of the history condition | 4.2 | `pics/fig_sharpness.tex` | `exp1b_boundary.dat`, `exp1d_tau.dat` | `python -m experiments.exp1_sharpness` |
| 3: effect of the delay at fixed `R0`; delay-free calibration | 4.3 | `pics/fig_delay_matters.tex` | `exp2a_delays.dat`, `exp2b_mismatch.dat` | `python -m experiments.exp2_delay_matters` |
| 4: loss of positivity at the Table 1 parameters; delay-induced oscillations | 4.3 | `pics/fig_positivity_realistic.tex` | `exp2c_positivity.dat`, `exp2d_hopf.dat` | `python -m experiments.exp2_delay_matters` |
| 5: Italian incidence with change-point fits of `beta(t - tau)` | 4.4 | `pics/fig_changepoint.tex` | `exp3c_fits.dat` | `python -m experiments.exp3_realdata` |

All data tables live in `pics/data/`. Figures 3 and 5 also read numbers from
`exp2_macros.tex` and `exp3_macros.tex` (the calibration window and the fitted
change point), so run the whole experiment rather than editing tables by hand.

### Numbers quoted in the text

Each experiment writes `pics/data/expN_macros.tex`. The manuscript loads these
files, so every number in Section 4 (convergence rates, critical thresholds,
fitted reproduction numbers, growth rates, change-point dates, …) comes from the
code. The remaining tables, `exp0_positivity_h.dat`, `exp1b_raster.dat`,
`exp1c_split.dat`, `exp1e_balance.dat` and `exp4b_jumps.dat`, are not plotted.
They contain the underlying values for statements in the text: discrete
positivity for every step size, the full positivity raster, the realistic
history split, the balance condition, and the derivative jumps at multiples of
`tau`.

## What each experiment computes

**exp0, verification (Section 4.1).** First-order convergence with no order
reduction, for the sub-critical admissible set and the COVID-19 set. Also checks
the discrete positivity and population bound of Theorem 3 for every step size
from `h = tau` down to `h = tau/1024`, across four delays.

**exp1, sharpness (Section 4.2).** The history condition
`N0_bar < 2 sqrt(Lambda/beta)` is attained for large delays and is very
conservative for short ones (Figure 2). With a realistic history split the
boundary follows `beta S0 I0 = Lambda`. The balance condition is not redundant:
above a critical `beta`, positivity fails for histories of any size.

**exp2, effect of the delay (Section 4.3).** At fixed `R0` the delay lowers the
growth rate and shifts and lowers the peak. A delay-free model calibrated to
early delay-model data fits that window essentially exactly but under-estimates
`R0` and the peak (Figure 3). At the COVID-19 parameters of Table 1 positivity
is lost, independently of the step size. A super-critical set satisfying the
hypotheses of Theorem 2 shows delay-induced oscillations (Figure 4).

**exp3, real data (Section 4.4).** Italian COVID-19 and Liberian Ebola growth
rates, the reproduction numbers they imply with and without the delay, and
piecewise-constant `beta(t - tau)` fits to the Italian incidence, once with a
fitted change point and once with the change point pinned to the lockdown
(Figure 5).

**exp4, limits and regularity (Section 4.1).** Convergence to the classical SIR
model at order one in `tau` (Figure 1, right), and the derivative jumps at
multiples of `tau`.

## Data

`python/data/` contains verbatim copies of two public data sets:

* **COVID-19, Italy**: Dipartimento della Protezione Civile,
  <https://github.com/pcm-dpc/COVID-19> (CC-BY-4.0).
* **Ebola, West Africa 2014**: compiled by C. Rivers from WHO and ministry of
  health situation reports, <https://github.com/cmrivers/ebola>.

`sirdelay/data.py` uses these cached copies. It downloads a file only if the
copy is missing, and it raises an error if the download fails; it never
substitutes other data. The source URLs, population figures used for
normalisation, SHA-256 checksums and caveats are in `python/data/PROVENANCE.md`.
Check a local copy against the checksums with

```sh
shasum -a 256 python/data/*.csv
```

## Directory layout requirement

The export code writes to `pics/data/` relative to the repository root. The
root is found as two levels above `python/sirdelay/export.py`, so keep
`python/` and `pics/` side by side.
