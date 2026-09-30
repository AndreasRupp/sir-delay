"""Download, cache and load the two real-world datasets used in experiment 3.

Nothing here is hand-entered or synthesised.  Both series are fetched from their
public upstream repositories, cached verbatim under ``python/data/`` and
checksummed; ``python/data/PROVENANCE.md`` records the URLs, licences and the
SHA-256 of the exact snapshots used.  If a download fails the loader raises
rather than falling back to invented numbers.

Sources
-------
COVID-19, Italy
    Dipartimento della Protezione Civile, "COVID-19 Italia - Monitoraggio
    situazione", national daily time series (CC-BY-4.0).
    https://github.com/pcm-dpc/COVID-19
    The columns map onto the SIR compartments almost directly:
    ``totale_positivi`` is the number of currently positive cases (-> I),
    ``dimessi_guariti + deceduti`` is the cumulative removed count (-> R) and
    ``nuovi_positivi`` is the daily incidence of newly reported cases.

Ebola virus disease, West Africa 2014
    WHO and national ministry situation reports, compiled by C. Rivers,
    "Data for the 2014 Ebola outbreak in West Africa".
    https://github.com/cmrivers/ebola
    Cumulative case and death counts per country; only the Liberia columns are
    used, matching the calibration of Rachah & Torres cited in the manuscript.

Population figures used to convert counts into the densities of the model:
Italy 59,641,488 (ISTAT, resident population 1 January 2020) and Liberia
4,396,873 (World Bank, 2014).  The estimated growth rate ``r`` -- and therefore
every reproduction number derived from it -- is invariant under this
normalisation; the population only enters the trajectory fits.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

ITALY_URL = ("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/"
             "dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
ITALY_FILE = "dpc-covid19-ita-andamento-nazionale.csv"
ITALY_POPULATION = 59_641_488          # ISTAT, 1 January 2020

EBOLA_URL = ("https://raw.githubusercontent.com/cmrivers/ebola/master/"
             "country_timeseries.csv")
EBOLA_FILE = "ebola_country_timeseries.csv"
LIBERIA_POPULATION = 4_396_873         # World Bank, 2014

# National lockdown, DPCM 9 March 2020 ("Io resto a casa"), in force 10 March.
ITALY_LOCKDOWN = pd.Timestamp("2020-03-09")

__all__ = [
    "DATA_DIR", "ITALY_LOCKDOWN", "ITALY_POPULATION", "LIBERIA_POPULATION",
    "fetch", "load_italy", "load_liberia", "sha256", "write_provenance",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, filename: str, *, refresh: bool = False) -> Path:
    """Return the cached copy of ``url``, downloading it once if necessary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    if path.exists() and not refresh:
        return path
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
    except Exception as exc:                        # pragma: no cover - network
        raise RuntimeError(
            f"could not download {url}.\n"
            f"Place the file manually at {path} and re-run; the experiments "
            f"never fall back to synthetic data."
        ) from exc
    path.write_bytes(payload)
    return path


def load_italy(*, refresh: bool = False) -> pd.DataFrame:
    """Daily national COVID-19 series for Italy as model densities.

    Returns columns ``date``, ``day`` (days since 24 February 2020), ``I``, ``R``,
    ``S``, ``incidence`` (all densities) and the raw counts.
    """
    path = fetch(ITALY_URL, ITALY_FILE, refresh=refresh)
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["data"]).dt.normalize(),
        "active": raw["totale_positivi"].astype(float),
        "removed": (raw["dimessi_guariti"] + raw["deceduti"]).astype(float),
        "new_cases": raw["nuovi_positivi"].astype(float),
        "cumulative": raw["totale_casi"].astype(float),
    }).sort_values("date").reset_index(drop=True)

    pop = ITALY_POPULATION
    df["day"] = (df["date"] - df["date"].iloc[0]).dt.days.astype(float)
    df["I"] = df["active"] / pop
    df["R"] = df["removed"] / pop
    df["S"] = 1.0 - df["I"] - df["R"]
    df["incidence"] = df["new_cases"] / pop
    df.attrs["source"] = ITALY_URL
    df.attrs["population"] = pop
    df.attrs["t0"] = df["date"].iloc[0]
    return df


def load_liberia(*, refresh: bool = False) -> pd.DataFrame:
    """Cumulative Ebola cases and deaths for Liberia, 2014-2015.

    The upstream file is in reverse chronological order and has gaps (situation
    reports were not daily); rows without a Liberia case count are dropped.
    """
    path = fetch(EBOLA_URL, EBOLA_FILE, refresh=refresh)
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], format="%m/%d/%Y"),
        "cumulative": raw["Cases_Liberia"].astype(float),
        "deaths": raw["Deaths_Liberia"].astype(float),
    })
    df = df.dropna(subset=["cumulative"]).sort_values("date").reset_index(drop=True)
    df["day"] = (df["date"] - df["date"].iloc[0]).dt.days.astype(float)
    df["cases_density"] = df["cumulative"] / LIBERIA_POPULATION
    df.attrs["source"] = EBOLA_URL
    df.attrs["population"] = LIBERIA_POPULATION
    df.attrs["t0"] = df["date"].iloc[0]
    return df


def write_provenance() -> Path:
    """Record URLs, licences and checksums of the cached snapshots."""
    italy = DATA_DIR / ITALY_FILE
    ebola = DATA_DIR / EBOLA_FILE
    text = f"""# Data provenance

Both files in this directory are downloaded verbatim by `sirdelay/data.py`
(`python -m experiments.exp3_realdata` fetches them on first run).  No values
are hand-entered, interpolated from figures, or synthesised.

## COVID-19, Italy

* File: `{ITALY_FILE}`
* URL: <{ITALY_URL}>
* Publisher: Dipartimento della Protezione Civile, Presidenza del Consiglio dei
  Ministri -- "COVID-19 Italia, Monitoraggio della situazione",
  <https://github.com/pcm-dpc/COVID-19>
* Licence: CC-BY-4.0
* SHA-256: `{sha256(italy) if italy.exists() else "(not downloaded)"}`
* Columns used: `data`, `totale_positivi` (currently positive -> I),
  `dimessi_guariti` + `deceduti` (cumulative removed -> R), `nuovi_positivi`
  (daily incidence), `totale_casi`.
* Series starts 24 February 2020.  Intervention date used in experiment 3:
  {ITALY_LOCKDOWN.date()} (DPCM "Io resto a casa", national lockdown).
* Population for the density normalisation: {ITALY_POPULATION:,} (ISTAT,
  resident population 1 January 2020).

## Ebola virus disease, West Africa 2014

* File: `{EBOLA_FILE}`
* URL: <{EBOLA_URL}>
* Publisher: compiled by C. Rivers from World Health Organization and national
  ministry of health situation reports, <https://github.com/cmrivers/ebola>
* SHA-256: `{sha256(ebola) if ebola.exists() else "(not downloaded)"}`
* Columns used: `Date`, `Cases_Liberia`, `Deaths_Liberia` (cumulative counts).
* Population for the density normalisation: {LIBERIA_POPULATION:,} (World Bank,
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
"""
    path = DATA_DIR / "PROVENANCE.md"
    path.write_text(text)
    return path
