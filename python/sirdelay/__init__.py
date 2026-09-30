"""Numerics for "Positivity of an SIR model with delay" (Altmann & Rupp).

The package implements the implicit-Euler scheme of Section 2.3 of the manuscript
for the delayed SIR model

    S'(t) = Lambda - beta * S(t-tau) * I(t-tau) - mu_S * S(t),
    I'(t) = beta * S(t-tau) * I(t-tau) - (gamma + mu_I) * I(t),
    R'(t) = gamma * I(t) - mu_R * R(t),

together with the analysis, fitting and export utilities used by the five
experiments in ``python/experiments``.
"""

from .parameters import ACADEMIC, COVID, EBOLA, Params
from .model import Solution, solve

__all__ = ["ACADEMIC", "COVID", "EBOLA", "Params", "Solution", "solve"]
