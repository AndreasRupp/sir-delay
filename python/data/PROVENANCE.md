# Data provenance

Both files in this directory are downloaded verbatim by `sirdelay/data.py`
(`python -m experiments.exp3_realdata` fetches them on first run).  No values
are hand-entered, interpolated from figures, or synthesised.

## COVID-19, Italy

* File: `dpc-covid19-ita-andamento-nazionale.csv`
* URL: <https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv>
* Publisher: Dipartimento della Protezione Civile, Presidenza del Consiglio dei
  Ministri -- "COVID-19 Italia, Monitoraggio della situazione",
  <https://github.com/pcm-dpc/COVID-19>
* Licence: CC-BY-4.0
* SHA-256: `30f6a1bd3aa3d99d9072af0e9be8ab23c7cab78e09651e1017b37ae0b97a050b`
* Columns used: `data`, `totale_positivi` (currently positive -> I),
  `dimessi_guariti` + `deceduti` (cumulative removed -> R), `nuovi_positivi`
  (daily incidence), `totale_casi`.
* Series starts 24 February 2020.  Intervention date used in experiment 3:
  2020-03-09 (DPCM "Io resto a casa", national lockdown).
* Population for the density normalisation: 59,641,488 (ISTAT,
  resident population 1 January 2020).

## Ebola virus disease, West Africa 2014

* File: `ebola_country_timeseries.csv`
* URL: <https://raw.githubusercontent.com/cmrivers/ebola/master/country_timeseries.csv>
* Publisher: compiled by C. Rivers from World Health Organization and national
  ministry of health situation reports, <https://github.com/cmrivers/ebola>
* SHA-256: `fda0a9972ec3bc827c7ccdae688acb85dc7c36581efee261c4bb439605d96be4`
* Columns used: `Date`, `Cases_Liberia`, `Deaths_Liberia` (cumulative counts).
* Population for the density normalisation: 4,396,873 (World Bank,
  Liberia 2014).

## Caveats

* Both series are *reported* cases, not infections.  Ascertainment is
  incomplete and time-varying, and reporting introduces its own lag which is
  confounded with the model delay `tau`.  Experiment 3 therefore estimates the
  exponential growth rate `r` from a pre-intervention window, where reporting
  effort is roughly constant, and states the confound explicitly rather than
  attempting to deconvolve it.
* The Ebola series is cumulative only and irregularly spaced, so it is used for
  the growth-rate and R0 comparison but not for the trajectory fit.
* Growth rates -- and hence every reproduction number derived from them -- are
  invariant under the population normalisation above.
