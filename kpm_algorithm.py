"""Die drei Wege, aus zensierten Zeiten eine Verweilzeit zu schätzen (je Weg eine Funktion, nur numpy über kpm_core).

* „Zensierte als Ereignis“: jede beobachtete Zeit zählt als Abholung zu diesem Zeitpunkt. Das ist dasselbe wie das Mittel aller beobachteten
  Zeiten und der untere Median aller beobachteten Zeiten (kleinste Zeit mit Ŝ ≤ 0,5): Kurve, Mittel und Median kommen aus denselben Zahlen (Befund gegenüber dem Plan, README).
* „Nur Ereignisse“: wer beim Ende der Beobachtung noch da ist, wird weggelassen.
* Kaplan–Meier: jeder Container zählt genau bis zu dem Tag, an dem seine Beobachtung endet; danach fehlt er in der Risikomenge.

Jeder Weg liefert eine Kurve (Treppe), einen Median und, wo er definiert ist, ein Mittel. Kaplan–Meier hat in der Regel kein Mittel der Verweilzeit: ist die
größte beobachtete Zeit zensiert, endet die Kurve oberhalb von null. Stattdessen gibt es die mittlere Lebenszeit bis τ (`restricted_mean_checked`)."""

from dataclasses import dataclass

import numpy as np

import kpm_core as K
from kpm_scenario import Sample

WAY_ALL = "Zensierte als Ereignis"
WAY_EVENTS = "Nur Ereignisse"
WAY_KM = "Kaplan–Meier"
WAYS = (WAY_ALL, WAY_EVENTS, WAY_KM)

_NAN = float("nan")
EMPTY_KM = {"t": np.zeros(0), "n": np.zeros(0), "d": np.zeros(0), "s": np.zeros(0), "gw": np.zeros(0), "n_obs": 0}


@dataclass(frozen=True)
class Way:
    name: str
    km: dict            # Treppe der Kurve (Arrays t, s, ...)
    median: float       # nan: nicht schätzbar (die Kurve erreicht 0,5 nicht) oder keine Ereignisse
    mean: float         # nan: für Kaplan–Meier nicht definiert
    status: str         # "ok", "nicht_schaetzbar" (Kurve erreicht 0,5 nicht) oder "keine_ereignisse"


def _status(km: dict, median: float) -> str:
    if km["t"].size == 0:
        return "keine_ereignisse"
    return "ok" if not np.isnan(median) else "nicht_schaetzbar"


def way_all(sample: Sample) -> Way:
    """Zensierte als Ereignis: Kaplan–Meier auf den beobachteten Zeiten, als wären alle Ereignisse; Mittel und Median der beobachteten Zeiten."""
    km = K.kaplan_meier(sample.obs, np.ones(sample.n, dtype=bool))
    med = K.quantile(km, 0.5)
    return Way(WAY_ALL, km, med, float(np.mean(sample.obs)), _status(km, med))


def way_events(sample: Sample) -> Way:
    """Nur Ereignisse: die Zensierten fehlen ganz; Mittel und Median der Zeiten, in denen eine Abholung zu sehen war."""
    if sample.n_events == 0:
        return Way(WAY_EVENTS, EMPTY_KM, _NAN, _NAN, "keine_ereignisse")
    t = sample.obs[sample.event]
    km = K.kaplan_meier(t, np.ones(t.size, dtype=bool))
    med = K.quantile(km, 0.5)
    return Way(WAY_EVENTS, km, med, float(np.mean(t)), _status(km, med))


def way_km(sample: Sample) -> Way:
    """Kaplan–Meier mit den Zensierten als „mindestens so lang“."""
    if sample.n_events == 0:
        return Way(WAY_KM, EMPTY_KM, _NAN, _NAN, "keine_ereignisse")
    km = K.kaplan_meier(sample.obs, sample.event)
    med = K.quantile(km, 0.5)
    return Way(WAY_KM, km, med, _NAN, _status(km, med))


def all_ways(sample: Sample) -> dict[str, Way]:
    return {WAY_ALL: way_all(sample), WAY_EVENTS: way_events(sample), WAY_KM: way_km(sample)}


def restricted_mean_checked(sample: Sample, tau: float) -> tuple[float, str]:
    """Mittlere Lebenszeit bis τ aus Kaplan–Meier mit Status: "ok", "tau_ueber_beobachtung" (τ liegt hinter der größten Beobachtungszeit, dort ist die
    Fläche nur fortgeschrieben) oder "keine_ereignisse". Kein stiller Wert in den Fehlerfällen."""
    if sample.n_events == 0:
        return _NAN, "keine_ereignisse"
    if tau > float(sample.obs.max()):
        return _NAN, "tau_ueber_beobachtung"
    return K.restricted_mean(K.kaplan_meier(sample.obs, sample.event), tau), "ok"
