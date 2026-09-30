"""Run every experiment in order and regenerate all figure data.

    python -m experiments.run_all

Writes ``pics/data/*.dat`` (pgfplots tables) and ``pics/data/*_macros.tex``
(numbers quoted in the manuscript text).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import (exp0_verification, exp1_sharpness,          # noqa: E402
                         exp2_delay_matters, exp3_realdata,
                         exp4_limit_regularity)
from sirdelay.export import DATA_OUT, banner                          # noqa: E402

EXPERIMENTS = (
    ("0  verification", exp0_verification),
    ("1  sharpness of the positivity conditions", exp1_sharpness),
    ("2  consequences of neglecting the delay", exp2_delay_matters),
    ("3  real-world data", exp3_realdata),
    ("4  tau -> 0 limit and regularity", exp4_limit_regularity),
)


def main() -> None:
    t_start = time.time()
    for label, module in EXPERIMENTS:
        banner(f"EXPERIMENT {label}")
        t0 = time.time()
        module.main()
        print(f"\n  [{label}] finished in {time.time() - t0:.1f} s")
    banner("all experiments complete")
    written = sorted(p.name for p in DATA_OUT.iterdir())
    print(f"  {len(written)} files in {DATA_OUT}:")
    for name in written:
        print(f"    {name}")
    print(f"\n  total wall time {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
