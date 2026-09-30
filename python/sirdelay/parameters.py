"""Parameter sets used in the manuscript.

``ACADEMIC`` is the dimensionless admissible set of Section 4.2 (main.tex:569-575);
``EBOLA`` and ``COVID`` are the literature-based sets of Table 1 (main.tex:511-533).
All rates are in units of 1/day, ``tau`` in days, and S, I, R are population
densities so that the disease-free state has N ~= 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    """Coefficients of the delayed SIR model."""

    Lambda: float
    beta: float
    gamma: float
    mu_S: float
    mu_I: float
    mu_R: float
    tau: float
    name: str = ""

    # -- derived quantities -------------------------------------------------
    @property
    def mu(self) -> float:
        """mu = min{mu_S, mu_I, mu_R} as in main.tex:285."""
        return min(self.mu_S, self.mu_I, self.mu_R)

    @property
    def S_star(self) -> float:
        """Disease-free susceptible level Lambda / mu_S."""
        return self.Lambda / self.mu_S

    @property
    def R0(self) -> float:
        """Basic reproduction number beta * S* / (gamma + mu_I), main.tex:472-476."""
        return self.beta * self.S_star / (self.gamma + self.mu_I)

    @property
    def threshold(self) -> float:
        """sqrt(Lambda / beta), the threshold appearing in Theorem 2."""
        return math.sqrt(self.Lambda / self.beta)

    def with_(self, **kwargs) -> "Params":
        """Return a copy with individual fields replaced."""
        return replace(self, **kwargs)

    # -- hypotheses of Theorem 2 -------------------------------------------
    def theorem2(self, N0_bar: float, factor: float = 2.0) -> dict:
        """Evaluate the hypotheses of Theorem 2 for a history with sup N = ``N0_bar``.

        ``factor = 2`` is Theorem 2 as stated in the manuscript,

            N0_bar < 2 sqrt(Lambda/beta)    and     Lambda * beta <= 4 mu**2,

        which rests on the arithmetic-geometric mean bound S*I <= N**2/4.
        ``factor = 1`` is the weaker variant that the cruder estimate
        S*I <= N**2 would give, retained only for comparison:

            N0_bar < sqrt(Lambda/beta)      and     Lambda * beta <= mu**2.
        """
        cond_history = N0_bar < factor * self.threshold
        cond_balance = self.Lambda * self.beta <= factor**2 * self.mu**2
        return {
            "factor": factor,
            "N0_bar": N0_bar,
            "threshold": factor * self.threshold,
            "cond_history": bool(cond_history),
            "cond_balance": bool(cond_balance),
            "satisfied": bool(cond_history and cond_balance),
            "population_bound": max(N0_bar, self.Lambda / self.mu),
        }


# Dimensionless example of Section 4.2 that satisfies the hypotheses of Theorem 2.
# tau is left free in the manuscript ("tau > 0 arbitrary"); 1.0 is only a default.
ACADEMIC = Params(
    Lambda=1.0, beta=0.8, gamma=0.3,
    mu_S=0.9, mu_I=0.9, mu_R=0.9,
    tau=1.0, name="academic",
)

# Table 1, Ebola (2014, West Africa).
EBOLA = Params(
    Lambda=2.7e-5, beta=0.2, gamma=0.03,
    mu_S=2.7e-5, mu_I=0.07, mu_R=2.7e-5,
    tau=10.0, name="ebola",
)

# Table 1, COVID-19 (early, Italy).
COVID = Params(
    Lambda=3.3e-5, beta=0.4, gamma=0.13,
    mu_S=3.3e-5, mu_I=6.0e-4, mu_R=3.3e-5,
    tau=5.0, name="covid",
)

ALL = {p.name: p for p in (ACADEMIC, EBOLA, COVID)}
