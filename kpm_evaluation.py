"""Kennzahlen, Verdikt-Logik und Auswertung über feste Zufalls-Seeds.

Alle Verdikte sind reine Funktionen (Art, Text) von Zahlen; die App ruft `getattr(st, Art)(Text)`, tests/test_verdicts.py prüft Schwellen und jede Zahl im Text."""

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

import kpm_algorithm as A
import kpm_constants as C
import kpm_core as K
import kpm_scenario as S

NAN = float("nan")
BAND_POINTS = (2, 4, 8, 12)   # Zeitpunkte, an denen die App prüft, ob die wahre Kurve im Band liegt


# ------------------------------------------------------------------------------------------------ Formate


def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma; nan als „nicht schätzbar“."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nicht schätzbar"
    return f"{x:.{nd}f}".replace(".", ",")


def signed(x: float, nd: int = 2) -> str:
    """Zahl mit Vorzeichen, Dezimalkomma und echtem Minuszeichen (Abweichungen in Einheiten der Größe)."""
    return f"{x:+.{nd}f}".replace(".", ",").replace("-", "−")


def pct(x: float, nd: int = 1) -> str:
    """Prozentzahl mit Vorzeichen, Dezimalkomma und echtem Minuszeichen."""
    return f"{x:+.{nd}f}".replace(".", ",").replace("-", "−") + " %"


def deviation_percent(value: float, truth: float) -> float:
    """Abweichung in Prozent der WAHRHEIT (Basis ist die Wahrheit, nicht die Schätzung); nan bleibt nan."""
    if value is None or math.isnan(value) or truth == 0:
        return NAN
    return 100.0 * (value - truth) / truth


def fraction_text(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


# ------------------------------------------------------------------------------------------------ Einstellungen und Analyse


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    window: int = C.DEFAULT_W
    kind: str = C.KIND_WINDOW
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    sample: S.Sample
    ways: dict
    censored_share: float
    true_median: float
    true_mean: float

    @property
    def km(self) -> A.Way:
        return self.ways[A.WAY_KM]


def analyse(settings: Settings) -> Analysis:
    sample = S.simulate(settings.kind, settings.window, settings.n, settings.seed)
    return Analysis(settings, sample, A.all_ways(sample), sample.censored_share, C.MEDIAN_TRUE, S.true_mean())


def lowest_survival(km: dict) -> float:
    """Niedrigster Wert der Kurve (1, wenn es kein Ereignis gibt)."""
    return float(km["s"][-1]) if km["s"].size else 1.0


def quantile_rows(a: Analysis, levels=(0.1, 0.25, 0.5)) -> list[dict]:
    """Quantile der Kaplan–Meier-Kurve gegen die wahren Quantile; nan = die Kurve erreicht das Niveau nicht."""
    rows = []
    km = a.km.km
    for q in levels:
        est = K.quantile(km, q) if km["t"].size else NAN
        rows.append({"q": q, "estimate": est, "truth": S.true_quantile(q)})
    return rows


# ------------------------------------------------------------------------------------------------ Wiederholung über feste Zufalls-Seeds


def repeat_summary(kind: str, window: int, n: int, seeds=C.BAND_SEEDS) -> dict:
    """Median und Mittel je Weg über die festen Zufalls-Seeds: je Weg Liste der Mediane und der Mittel (nan bleibt nan)."""
    out = {name: {"median": [], "mean": []} for name in A.WAYS}
    shares = []
    for seed in seeds:
        sample = S.simulate(kind, window, n, seed)
        shares.append(sample.censored_share)
        for name, way in A.all_ways(sample).items():
            out[name]["median"].append(way.median)
            out[name]["mean"].append(way.mean)
    return {"seeds": len(seeds), "ways": out, "censored_share": float(np.mean(shares))}


def spread(values) -> dict:
    """Median, Minimum und Maximum der definierten Werte, dazu wie viele definiert sind."""
    a = np.asarray(values, dtype=float)
    ok = a[~np.isnan(a)]
    if ok.size == 0:
        return {"median": NAN, "low": NAN, "high": NAN, "defined": 0, "of": int(a.size)}
    return {"median": float(np.median(ok)), "low": float(ok.min()), "high": float(ok.max()), "defined": int(ok.size), "of": int(a.size)}


# ------------------------------------------------------------------------------------------------ Greenwood-Band


def band_checks(a: Analysis, method: str, level: float = C.LEVEL, points=BAND_POINTS) -> list[dict]:
    """An den Zeitpunkten: wahres S(t), geschätztes S(t), Band, liegt die Wahrheit im Band? Zeitpunkte hinter der größten Beobachtungszeit fehlen."""
    rows = []
    if a.km.km["t"].size == 0:
        return rows
    tmax = float(a.sample.obs.max())
    for t in points:
        if t > tmax:
            continue
        s, lo, hi = K.greenwood_band(a.km.km, t, level, method)
        truth = float(S.true_survival(t))
        inside = None if math.isnan(lo) else bool(lo <= truth <= hi)
        rows.append({"t": t, "truth": truth, "s": s, "low": lo, "high": hi, "inside": inside})
    return rows


@lru_cache(maxsize=1)
def measurements() -> dict:
    """Eingefrorene Messung (tools/measure_kaplan_meier.py); nicht lesbar -> leeres Ergebnis, die App zeigt dann einen Hinweis statt einer Tabelle."""
    path = Path(__file__).resolve().parent / C.MEASURE_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def coverage_range(method: str) -> tuple[float, float, int] | None:
    """Kleinste und größte gemessene Überdeckung (Prozent) der Methode über alle Zellen der Messung, dazu die Zahl der Läufe je Zelle."""
    m = measurements()
    if not m.get("coverage"):
        return None
    values = [row[method] for block in m["coverage"] for row in block["rows"]]
    return min(values), max(values), int(m["coverage"][0]["runs"])


# ------------------------------------------------------------------------------------------------ Handbeispiel mit Brüchen


def hand_rows(times=S.HAND_TIMES, events=S.HAND_EVENTS) -> list[dict]:
    """Kaplan–Meier mit exakten Brüchen (die Handrechnung): je Ereigniszeit t, im Risiko n, Abholungen d, Faktor (n − d)/n und S(t).
    Konvention: bei gleicher Zeit zählt das Ereignis vor der Zensierung, der Zensierte bleibt im Risiko (Risikomenge: Zeit >= t)."""
    rows, s = [], Fraction(1)
    for t in sorted({x for x, e in zip(times, events) if e}):
        n = sum(1 for x in times if x >= t)
        d = sum(1 for x, e in zip(times, events) if x == t and e)
        factor = Fraction(n - d, n)
        s *= factor
        rows.append({"t": t, "n": n, "d": d, "factor": factor, "s": s})
    return rows


def hand_summary(times=S.HAND_TIMES, events=S.HAND_EVENTS) -> dict:
    """Median, Mittel aller und nur der Ereignisse, mittlere Lebenszeit bis zur größten Zeit und Greenwood-Varianz am Median, alles mit Brüchen."""
    rows = hand_rows(times, events)
    half = Fraction(1, 2)
    median = next((r["t"] for r in rows if r["s"] <= half), None)
    tau = max(times)
    area, prev_t, prev_s = Fraction(0), 0, Fraction(1)
    for r in rows:
        area += prev_s * (r["t"] - prev_t)
        prev_t, prev_s = r["t"], r["s"]
    area += prev_s * (tau - prev_t)
    var = None
    if median is not None:
        s_med = next(r["s"] for r in rows if r["t"] == median)
        gw = sum(Fraction(r["d"], r["n"] * (r["n"] - r["d"])) for r in rows if r["t"] <= median and r["n"] > r["d"])
        var = s_med**2 * gw
    return {
        "median": median,
        "mean_all": Fraction(sum(times), len(times)),
        "mean_events": Fraction(sum(x for x, e in zip(times, events) if e), sum(1 for e in events if e)),
        "rmst": area,
        "tau": tau,
        "greenwood_var": var,
    }


# ------------------------------------------------------------------------------------------------ Verdikte (reine Funktionen)


def verdict_hand(step: int, rows: list[dict], total_events: int) -> tuple[str, str]:
    """Verdikt des Handbeispiels nach `step` verarbeiteten Ereigniszeiten (0 bis Zahl der Ereigniszeiten)."""
    if step <= 0:
        return "info", "Noch nichts verarbeitet: Solange keine Abholung gesehen wurde, sind alle sechs Container im Risiko und S(t) = 1."
    row = rows[step - 1]
    text = (f"Nach t = {row['t']}: {row['n']} Container im Risiko, {row['d']} Abholung, Faktor {fraction_text(row['factor'])}, "
            f"S({row['t']}) = {fraction_text(row['s'])} = {de(float(row['s']), 4)}.")
    if step >= len(rows):
        text += " Das war die letzte Abholung: die Kurve ist am Ende."
    elif row["s"] <= Fraction(1, 2):
        text += " Die Kurve liegt jetzt bei höchstens 0,5: Dieses t ist der Median."
    return "info", text


def verdict_core(a: Analysis) -> tuple[str, str]:
    """Kernfrage: Wie weit liegen die Wege daneben? Nennt nur Zahlen dieser einen Stichprobe (Basis der Prozente ist die Wahrheit)."""
    if a.sample.n_events == 0:
        return "warning", ("Es wurde keine einzige Abholung beobachtet: Kaplan–Meier hat nichts zu schätzen, und das Mittel der beobachteten Zeiten ist nur "
                           f"die mittlere Beobachtungsdauer ({de(a.ways[A.WAY_ALL].mean)} Tage). Hier hilft nur eine längere Beobachtung oder mehr Container.")
    if a.sample.n_events == a.sample.n:
        return "info", ("Nichts ist zensiert: Alle Container wurden bis zur Abholung beobachtet, und die drei Wege liefern dieselben Zahlen "
                        f"(Median {de(a.km.median)} Tage, Mittel {de(a.ways[A.WAY_ALL].mean)} Tage).")
    share = f"{100 * a.censored_share:.1f}".replace(".", ",")
    allw, evw, kmw = a.ways[A.WAY_ALL], a.ways[A.WAY_EVENTS], a.km
    mean_dev = deviation_percent(allw.mean, a.true_mean)
    head = (f"{share} % der Container sind beim Ende der Beobachtung noch da. Das Mittel der beobachteten Zeiten liegt bei {de(allw.mean)} Tagen "
            f"(wahres Mittel {de(a.true_mean)}: {pct(mean_dev)} gegen die Wahrheit). ")
    if kmw.status == "nicht_schaetzbar":
        return "warning", head + (f"Die Kaplan–Meier-Kurve erreicht 0,5 nicht (niedrigster Wert {de(lowest_survival(kmw.km), 3)}): "
                                  f"Der Median ist hier nicht schätzbar. Als Ereignis gelesen ergäben die Zensierten einen Median von {de(allw.median)} Tagen, "
                                  f"wahr sind {de(a.true_median)}.")
    devs = {name: deviation_percent(w.median, a.true_median) for name, w in ((A.WAY_ALL, allw), (A.WAY_EVENTS, evw), (A.WAY_KM, kmw))}
    body = (f"Median in dieser Stichprobe: Zensierte als Ereignis {de(allw.median)} ({pct(devs[A.WAY_ALL])}), nur Ereignisse {de(evw.median)} ({pct(devs[A.WAY_EVENTS])}), "
            f"Kaplan–Meier {de(kmw.median)} ({pct(devs[A.WAY_KM])}); wahrer Median {de(a.true_median)} Tage.")
    kind = "success" if abs(devs[A.WAY_KM]) < min(abs(devs[A.WAY_ALL]), abs(devs[A.WAY_EVENTS])) else "info"
    return kind, head + body


def verdict_ways(rep: dict, true_median: float, true_mean: float) -> tuple[str, str]:
    """Experiment „Wege gegen die Wahrheit“ über die festen Zufalls-Seeds: Median der Schätzungen mit Spanne, nicht definierte Läufe genannt."""
    ways = rep["ways"]
    sm = {name: spread(ways[name]["median"]) for name in A.WAYS}
    mean_all = spread(ways[A.WAY_ALL]["mean"])
    km = sm[A.WAY_KM]
    parts = []
    for name in (A.WAY_ALL, A.WAY_EVENTS, A.WAY_KM):
        s = sm[name]
        if s["defined"] == 0:
            parts.append(f"{name}: in keinem der {s['of']} Läufe schätzbar")
        else:
            parts.append(f"{name} {de(s['median'])} (Spanne {de(s['low'])} bis {de(s['high'])})")
    text = (f"Median der Medianschätzungen über {rep['seeds']} feste Zufalls-Seeds (wahrer Median {de(true_median)}): " + "; ".join(parts) + ". "
            f"Das Mittel aller beobachteten Zeiten liegt im Median der Läufe bei {de(mean_all['median'])} (wahres Mittel {de(true_mean)}, "
            f"{pct(deviation_percent(mean_all['median'], true_mean))} gegen die Wahrheit).")
    if km["defined"] < km["of"]:
        text += f" Kaplan–Meier hat in {km['of'] - km['defined']} von {km['of']} Läufen keinen Median (Kurve erreicht 0,5 nicht); die Zahl oben gilt nur für die übrigen {km['defined']}."
        return "warning", text
    return "info", text


def verdict_band(method_label: str, rows: list[dict], cover: tuple[float, float, int] | None) -> tuple[str, str]:
    """Experiment Greenwood-Band: liegt die Wahrheit in dieser Stichprobe im Band, dazu die gemessene Überdeckung."""
    checked = [r for r in rows if r["inside"] is not None]
    if not checked:
        return "warning", "Für diese Stichprobe gibt es an keinem der Zeitpunkte ein Band (die Kurve liegt dort bei 1 oder 0 oder hinter der Beobachtung)."
    hit = sum(1 for r in checked if r["inside"])
    times = ", ".join(str(r["t"]) for r in checked)
    text = f"In dieser Stichprobe liegt die wahre Kurve an {hit} von {len(checked)} Zeitpunkten (t = {times}) im {method_label}-Band."
    if cover is not None:
        lo, hi, runs = cover
        text += (f" Gemessene Überdeckung bei nominal 95 % ({runs} Läufe je Zelle, alle Zellen der Tabelle unten): {de(lo, 1)} bis {de(hi, 1)} % "
                 f"für das {method_label}-Band; sie hängt an n, t und der Art des Bandes.")
    return ("success" if hit == len(checked) else "info"), text


def verdict_median(a: Analysis, rows: list[dict], defined_pct: float | None) -> tuple[str, str]:
    """Experiment „Median nicht schätzbar“: was in dieser Stichprobe geht und was nicht."""
    km = a.km
    if km.status == "keine_ereignisse":
        return "warning", "Keine Abholung beobachtet: weder Median noch ein anderes Quantil lässt sich schätzen."
    extra = ""
    if defined_pct is not None:
        extra = f" In der Messung (Fenster {a.settings.window} Tage, n = 300) war der Median in {de(defined_pct, 1)} % der Läufe schätzbar."
    if km.status == "nicht_schaetzbar":
        usable = [r for r in rows if not math.isnan(r["estimate"])]
        if usable:
            best = usable[-1]
            tail = (f" Stattdessen geht das {int(round(100 * best['q']))} %-Quantil: {de(best['estimate'])} Tage (wahr {de(best['truth'])}), "
                    "und die mittlere Lebenszeit bis τ (unten).")
        else:
            tail = " Auch kein tieferes Quantil ist schätzbar; es bleibt die mittlere Lebenszeit bis τ (unten) oder eine längere Beobachtung."
        return "warning", (f"Die Kurve erreicht 0,5 nicht (niedrigster Wert {de(lowest_survival(km.km), 3)}): der Median ist nicht schätzbar." + tail + extra)
    return "success", f"Die Kurve fällt auf höchstens 0,5: Median {de(km.median)} Tage (wahr {de(a.true_median)})." + extra


# ------------------------------------------------------------------------------------------------ Sätze aus der eingefrorenen Messung


def _join(values: list[str]) -> str:
    return ", ".join(values[:-1]) + " und " + values[-1] if len(values) > 1 else values[0]


def coverage_sentence() -> str:
    """Spanne der gemessenen Überdeckung je Bandart, dazu die Zelle mit dem kleinsten Wert des einfachen Bandes."""
    m = measurements()
    ll, pl = coverage_range("loglog"), coverage_range("plain")
    if not (ll and pl):
        return "Messung fehlt"
    worst = min(((row["plain"], block["length"], block["n"], row["t"]) for block in m["coverage"] for row in block["rows"]))
    return (f"log-log {de(ll[0], 1)} bis {de(ll[1], 1)} %, einfach {de(pl[0], 1)} bis {de(pl[1], 1)} % "
            f"(kleinster Wert des einfachen Bandes: {de(worst[0], 1)} % bei Fenster {worst[1]} Tage, n = {worst[2]}, t = {worst[3]}).")


def cells_sentence(meas: dict) -> str:
    """Satz zur Tabelle „Fenster und feste Frist“: die Zahlen stehen in der Messung, die Bedingung des Medians bei fester Frist in der Aussage selbst."""
    if not meas.get("cells"):
        return ""
    fixed = [c for c in meas["cells"] if c["kind"] == C.KIND_FIXED]
    window = [c for c in meas["cells"] if c["kind"] == C.KIND_WINDOW]

    def vals(cells, key):
        return _join([de(c[key]["mean"]) for c in cells])

    lengths = lambda cells: _join([str(c["length"]) for c in cells])  # noqa: E731
    return (f"Feste Frist {lengths(fixed)} Tage: Median von „Zensierte als Ereignis“ {vals(fixed, 'median_all')} (wahr {de(meas['meta']['median_true'])}), Mittel aller {vals(fixed, 'mean_all')} "
            f"(wahr {de(meas['meta']['mean_true'])}). Der Median stimmt bei einer Frist über dem wahren Median, weil dann mehr als die Hälfte der Container vor der Frist abgeholt wird; das Mittel stimmt nicht. "
            f"Fenster {lengths(window)} Tage: derselbe Median liegt bei {vals(window, 'median_all')}.")
