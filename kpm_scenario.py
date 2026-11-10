"""Szenario: Containerverweilzeit D lognormal (Median 4 Tage, σ = 0,9) und zwei Arten, sie zu beobachten.

Die Erzeuger sind reine Funktionen von (Art, Länge, n, Zufalls-Seed); der Zufall kommt nur aus SplitMix64 (kpm_core).
Die Wahrheit (Überlebensfunktion, Quantile, Mittel, mittlere Lebenszeit bis τ) steht in geschlossener Form hier, nicht in der Auswertung."""

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

import kpm_constants as C
from kpm_core import SplitMix64

_MU = math.log(C.MEDIAN_TRUE)
_NORMAL = NormalDist()


@dataclass(frozen=True)
class Sample:
    """Beobachtete Zeiten (Tage) und Ereignis (True = Abholung gesehen, False = beim Ende der Beobachtung noch da)."""

    obs: np.ndarray
    event: np.ndarray

    @property
    def n(self) -> int:
        return int(self.obs.size)

    @property
    def n_events(self) -> int:
        return int(self.event.sum())

    @property
    def censored_share(self) -> float:
        return 1.0 - self.n_events / self.n


def simulate(kind: str, length: float, n: int, seed: int) -> Sample:
    """Eine Stichprobe von n Containern.
    Fenster: Ankunft gleichverteilt in [0, length), die Beobachtung endet bei `length`, die Beobachtungsdauer ist also length − Ankunft.
    Feste Frist (Typ I): jeder Container wird genau `length` Tage lang beobachtet.
    Beobachtet wird min(D, Dauer); ein Ereignis ist D <= Dauer."""
    if kind not in C.KINDS:
        raise ValueError(f"Unbekannte Zensierungsart: {kind!r}")
    if length <= 0 or n < 1:
        raise ValueError("Länge muss größer als 0 und n mindestens 1 sein.")
    rng = SplitMix64(seed)
    d = rng.lognormal(n, C.MEDIAN_TRUE, C.SIGMA_TRUE)
    if kind == C.KIND_WINDOW:
        follow = float(length) - rng.uniform(n) * float(length)   # in (0, length]
    else:
        follow = np.full(n, float(length))
    return Sample(np.minimum(d, follow), d <= follow)


# ------------------------------------------------------------------------------------------------ Wahrheit (geschlossene Formeln)


def true_survival(t):
    """S(t) = 1 − Φ((ln t − ln Median)/σ); S(t) = 1 für t <= 0."""
    t = np.asarray(t, dtype=float)
    z = np.where(t > 0, (np.log(np.where(t > 0, t, 1.0)) - _MU) / C.SIGMA_TRUE, -np.inf)
    return 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2.0))


def true_quantile(q: float) -> float:
    """Wahres q-Quantil der Verweilzeit (q = 0,5 ergibt den Median 4)."""
    return math.exp(_MU + C.SIGMA_TRUE * _NORMAL.inv_cdf(q))


def true_mean() -> float:
    return math.exp(_MU + C.SIGMA_TRUE**2 / 2)


def true_restricted_mean(tau: float) -> float:
    """∫₀^τ S(t) dt = τ·S(τ) + E[D·1(D <= τ)], mit E[D·1(D <= τ)] = Mittel · Φ((ln τ − μ − σ²)/σ)."""
    if tau <= 0:
        return 0.0
    surv = float(true_survival(tau))
    partial = true_mean() * _NORMAL.cdf((math.log(tau) - _MU - C.SIGMA_TRUE**2) / C.SIGMA_TRUE)
    return tau * surv + partial


# ------------------------------------------------------------------------------------------------ Handbeispiel


HAND_TIMES = (2, 3, 4, 5, 5, 8)
HAND_EVENTS = (1, 0, 1, 1, 0, 1)    # 3+ und 5+ sind beim Ende der Beobachtung noch da
