"""Kern der Demo: Kaplan–Meier, Greenwood-Band, mittlere Lebenszeit bis τ, Zufallsstrom SplitMix64.

Kopie aus ueberlebenszeit-planung/kern/uz_core.py, Stand 2026-11-10, gekürzt auf das, was diese Demo braucht
(ohne Eintrittszeit, Vergleich zweier Gruppen, konkurrierende Risiken, Weibull und Cox). Geändert gegenüber der Vorlage:
`check_data` und `kaplan_meier` ohne Eintritt, die Zahl der Ereignisse je Zeit kommt aus `np.unique(..., return_counts=True)`;
neu ist nur `band_curve` (das Band an allen Ereigniszeiten in einem Zug, gegen `greenwood_band` im Test geprüft).

Konventionen: Risikomenge bei t = {i : Zeit_i >= t}; bei gleicher Zeit zählt das Ereignis vor der Zensierung (Zensierte bleiben im
Risiko); Zensierung gilt als unabhängig vom Ereignis. Zufall der Stichproben nur über SplitMix64, nicht über `hash()` oder numpy-Verteilungen."""
from __future__ import annotations

import math

import numpy as np

# ------------------------------------------------------------------------------------------------ Zufall

_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


class SplitMix64:
    """SplitMix64-Strom (Steele, Lea, Flood 2014). Ausgabe k hängt nur von (Seed, k) ab: plattform- und versionsunabhängig.
    Die Verteilungen sind hier selbst aus Gleichverteilten gebaut (nie numpy-Verteilungen, deren Algorithmen sich zwischen Versionen ändern)."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, size: int) -> np.ndarray:
        """Standardnormal (Box-Muller, nur der Kosinus-Zweig); verbraucht 2·size Gleichverteilte."""
        u1 = self.uniform(size)
        u2 = self.uniform(size)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)

    def lognormal(self, size: int, median: float, sigma: float) -> np.ndarray:
        """Lognormal mit gegebenem Median und σ der Log-Skala."""
        return median * np.exp(sigma * self.normal(size))


# ------------------------------------------------------------------------------------------------ Eingabeprüfung


class SurvivalError(ValueError):
    """Ungültige Eingabe (verständliche Meldung für die App)."""


def check_data(time, event):
    """Prüft und wandelt in float-/bool-Arrays. Zeiten > 0 und endlich, Ereignis 0/1."""
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise SurvivalError("Es braucht mindestens eine Beobachtung (eindimensionale Liste der Zeiten).")
    e_raw = np.asarray(event)
    if e_raw.shape != t.shape:
        raise SurvivalError("Zeiten und Ereignisse müssen gleich viele Einträge haben.")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise SurvivalError("Alle Zeiten müssen endlich und größer als 0 sein.")
    if not np.all(np.isin(e_raw, (0, 1, False, True))):
        raise SurvivalError("Das Ereignis muss 0 (zensiert) oder 1 (Ereignis) sein.")
    return t, e_raw.astype(bool)


# ------------------------------------------------------------------------------------------------ Kaplan–Meier


def kaplan_meier(time, event) -> dict:
    """Kaplan–Meier-Schätzer. Gibt Arrays je Ereigniszeit zurück:
    t, n (im Risiko), d (Ereignisse), s (Überleben nach t), gw (Greenwood-Summe Σ d/(n(n−d)), Terme mit n = d fehlen: dort ist s = 0), n_obs."""
    t, e = check_data(time, event)
    ts, d = np.unique(t[e], return_counts=True)
    n = (t.size - np.searchsorted(np.sort(t), ts, side="left")).astype(float)  # Anzahl mit Zeit >= ts
    d = d.astype(float)
    s = np.cumprod(1.0 - d / n) if ts.size else np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(n > d, d / (n * (n - d)), 0.0)
    return {"t": ts, "n": n, "d": d, "s": s, "gw": np.cumsum(terms), "n_obs": int(t.size)}


def _index_at(km: dict, x: float) -> int:
    return int(np.searchsorted(km["t"], x, side="right")) - 1


def survival_at(km: dict, x: float) -> float:
    """Ŝ(x); vor der ersten Ereigniszeit 1."""
    i = _index_at(km, x)
    return 1.0 if i < 0 else float(km["s"][i])


def quantile(km: dict, q: float) -> float:
    """Kleinste Ereigniszeit mit Ŝ ≤ 1 − q; nan, wenn die Kurve das Niveau nicht erreicht („nicht schätzbar“)."""
    hit = np.where(km["s"] <= 1.0 - q + 1e-12)[0]
    return float(km["t"][hit[0]]) if hit.size else float("nan")


def z_two_sided(level: float) -> float:
    """Normalquantil für ein zweiseitiges Niveau (Bisektion über die Fehlerfunktion, ohne scipy)."""
    if not 0.0 < level < 1.0:
        raise SurvivalError("Das Niveau muss zwischen 0 und 1 liegen.")
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if math.erf(mid / math.sqrt(2.0)) < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def greenwood_band(km: dict, x: float, level: float = 0.95, method: str = "loglog"):
    """Punktweises Konfidenzband an der Zeit x: (Ŝ, untere, obere Grenze) oder (Ŝ, nan, nan), wenn Ŝ ∈ {0, 1}.
    method „loglog“ (bleibt in [0, 1]) oder „plain“ (Ŝ ± z·SE, auf [0, 1] gekappt)."""
    if method not in ("loglog", "plain"):
        raise SurvivalError("method muss „loglog“ oder „plain“ sein.")
    i = _index_at(km, x)
    if i < 0:
        return 1.0, float("nan"), float("nan")
    s, gw = float(km["s"][i]), float(km["gw"][i])
    if s <= 0.0 or s >= 1.0:
        return s, float("nan"), float("nan")
    z = z_two_sided(level)
    if method == "plain":
        se = s * math.sqrt(gw)
        return s, max(0.0, s - z * se), min(1.0, s + z * se)
    se_ll = math.sqrt(gw) / abs(math.log(s))
    return s, s ** math.exp(z * se_ll), s ** math.exp(-z * se_ll)


def band_curve(km: dict, level: float = 0.95, method: str = "loglog"):
    """Das Band an allen Ereigniszeiten in einem Zug: (untere, obere) als Arrays, nan wo Ŝ = 0 (dasselbe wie `greenwood_band` je Zeit)."""
    if method not in ("loglog", "plain"):
        raise SurvivalError("method muss „loglog“ oder „plain“ sein.")
    z = z_two_sided(level)
    s, gw = km["s"], km["gw"]
    ok = (s > 0.0) & (s < 1.0)
    lo = np.full(s.shape, np.nan)
    hi = np.full(s.shape, np.nan)
    if method == "plain":
        se = np.where(ok, s * np.sqrt(gw), 0.0)
        lo[ok] = np.maximum(0.0, s[ok] - z * se[ok])
        hi[ok] = np.minimum(1.0, s[ok] + z * se[ok])
    else:
        se_ll = np.where(ok, np.sqrt(gw) / np.abs(np.log(np.where(ok, s, 0.5))), 0.0)
        lo[ok] = s[ok] ** np.exp(z * se_ll[ok])
        hi[ok] = s[ok] ** np.exp(-z * se_ll[ok])
    return lo, hi


def restricted_mean(km: dict, tau: float) -> float:
    """Mittlere Lebenszeit bis tau: Fläche unter der KM-Treppe (S = 1 bis zur ersten Ereigniszeit; hinter der letzten Beobachtung konstant fortgesetzt,
    deshalb darf tau die größte Beobachtungszeit nicht übersteigen, das prüft die Auswertung der App)."""
    if tau <= 0:
        raise SurvivalError("τ muss größer als 0 sein.")
    t, s = km["t"], km["s"]
    sel = t < tau
    edges = np.concatenate([[0.0], t[sel], [tau]])
    levels = np.concatenate([[1.0], s[sel]])
    return float(np.sum(levels * np.diff(edges)))
