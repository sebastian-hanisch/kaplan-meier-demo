"""Messungen der Demo (feste Seeds, SplitMix64): alle Zahlen der README und der vorberechneten Tabellen der App.

Aufruf (aus dem Repo-Ordner):  _venvs/test/Scripts/python.exe -X utf8 tools/measure_kaplan_meier.py
Schreibt tools/messung_kaplan_meier.json (eingefroren, im Repo) und gibt eine Textfassung aus. Laufzeit: wenige Sekunden bis eine Minute.

Welt: Verweilzeit lognormal, Median 4 Tage, σ = 0,9. Lauf r einer Messreihe mit Basis b benutzt den Zufalls-Seed b + r (Zensierungsmessung: b = 2000, Überdeckung: 5000, mittlere Lebenszeit: 6000).
Mittelwerte stehen mit Standardfehler des Mittels über die Läufe (ohne nicht definierte Läufe); `undef` ist der Anteil der Läufe ohne Median.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kpm_algorithm as A  # noqa: E402
import kpm_constants as C  # noqa: E402
import kpm_core as K  # noqa: E402
import kpm_scenario as S  # noqa: E402

RUNS = 1000
COVER_RUNS = 1000
BASE = 2000     # alle Zellen der Zensierungsmessung benutzen dieselben Zufalls-Seeds 2000 bis 2999 (die Verweilzeiten sind in Zellen mit gleichem n dieselben)
OUT = ROOT / "tools" / "messung_kaplan_meier.json"


def ms(values) -> dict:
    a = np.asarray(values, dtype=float)
    ok = a[~np.isnan(a)]
    if ok.size == 0:
        return {"mean": None, "se": None, "defined": 0, "of": int(a.size)}
    se = float(ok.std(ddof=1) / math.sqrt(ok.size)) if ok.size > 1 else None
    return {"mean": round(float(ok.mean()), 6), "se": None if se is None else round(se, 6), "defined": int(ok.size), "of": int(a.size)}


def cell(kind: str, length: int, n: int, base: int, runs: int = RUNS) -> dict:
    rows = {k: [] for k in ("censored", "mean_all", "mean_events", "median_all", "median_events", "median_km")}
    for r in range(runs):
        smp = S.simulate(kind, length, n, base + r)
        w = A.all_ways(smp)
        rows["censored"].append(smp.censored_share)
        rows["mean_all"].append(w[A.WAY_ALL].mean)
        rows["mean_events"].append(w[A.WAY_EVENTS].mean)
        rows["median_all"].append(w[A.WAY_ALL].median)
        rows["median_events"].append(w[A.WAY_EVENTS].median)
        rows["median_km"].append(w[A.WAY_KM].median)
    out = {"kind": kind, "length": length, "n": n, "base": base, "runs": runs}
    out.update({k: ms(v) for k, v in rows.items()})
    out["undef_km"] = round(1.0 - out["median_km"]["defined"] / runs, 4)
    return out


def coverage(length: int, n: int, times, base: int, runs: int = COVER_RUNS) -> dict:
    hits = {(m, t): 0 for m in ("loglog", "plain") for t in times}
    undef = {(m, t): 0 for m in ("loglog", "plain") for t in times}
    truth = {t: float(S.true_survival(t)) for t in times}
    for r in range(runs):
        smp = S.simulate(C.KIND_WINDOW, length, n, base + r)
        km = K.kaplan_meier(smp.obs, smp.event)
        for m in ("loglog", "plain"):
            for t in times:
                s, lo, hi = K.greenwood_band(km, t, C.LEVEL, m)
                if math.isnan(lo):
                    undef[(m, t)] += 1
                elif lo <= truth[t] <= hi:
                    hits[(m, t)] += 1
    rows = []
    for t in times:
        row = {"t": t, "true_s": round(truth[t], 6)}
        for m in ("loglog", "plain"):
            row[m] = round(100.0 * hits[(m, t)] / runs, 1)
            row[m + "_undef"] = undef[(m, t)]
        rows.append(row)
    return {"kind": C.KIND_WINDOW, "length": length, "n": n, "base": base, "runs": runs, "rows": rows}


def rmst_cell(length: int, tau: float, n: int, base: int, runs: int = RUNS) -> dict:
    est, naive = [], []
    for r in range(runs):
        smp = S.simulate(C.KIND_WINDOW, length, n, base + r)
        v, st = A.restricted_mean_checked(smp, tau)
        est.append(v)
        naive.append(float(np.mean(np.minimum(smp.obs, tau))))
    return {"length": length, "tau": tau, "n": n, "base": base, "runs": runs, "true": round(S.true_restricted_mean(tau), 6),
            "km": ms(est), "naive_min_obs": ms(naive)}


def main() -> dict:
    res = {"meta": {"runs": RUNS, "cover_runs": COVER_RUNS, "median_true": C.MEDIAN_TRUE, "sigma": C.SIGMA_TRUE,
                    "mean_true": round(S.true_mean(), 6), "q90_true": round(S.true_quantile(0.9), 6)}}
    res["v1"] = [cell(C.KIND_WINDOW, 30, n, BASE) for n in (100, 300, 1000)]
    res["cells"] = [cell(kind, w, 300, BASE) for kind in C.KINDS for w in (30, 8, 4)]
    res["defined"] = [cell(C.KIND_WINDOW, w, 300, BASE) for w in (4, 5, 6, 8, 12, 30)]
    res["defined_n"] = [cell(C.KIND_WINDOW, 4, n, BASE) for n in (100, 300, 1000)]
    res["coverage"] = [coverage(30, 100, (2, 4, 8, 12), 5000), coverage(30, 300, (2, 4, 8, 12), 5000), coverage(8, 300, (2, 4, 6), 5000)]
    res["rmst"] = [rmst_cell(8, 6, 300, 6000), rmst_cell(30, 12, 300, 6000)]
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return res


def fmt(d) -> str:
    if d["mean"] is None:
        return "n/a"
    return f"{d['mean']:.2f} ± {d['se']:.2f}" if d["se"] is not None else f"{d['mean']:.2f}"


if __name__ == "__main__":
    r = main()
    print(f"Wahrheit: Median {r['meta']['median_true']}, Mittel {r['meta']['mean_true']:.3f}, Quantil 0,9 {r['meta']['q90_true']:.3f}; Läufe {RUNS}, Läufe der Überdeckung {COVER_RUNS}")
    for block in ("v1", "cells", "defined", "defined_n"):
        print(f"\n== {block}")
        for c in r[block]:
            print(f"{c['kind'][:10]:10s} L={c['length']:3d} n={c['n']:5d} zensiert {100*c['censored']['mean']:5.1f} %  "
                  f"Mittel alle {fmt(c['mean_all'])}  nur Ereignisse {fmt(c['mean_events'])}  | Median alle {fmt(c['median_all'])}  "
                  f"nur Ereignisse {fmt(c['median_events'])}  KM {fmt(c['median_km'])}  (KM undefiniert {100*c['undef_km']:.1f} %)")
    print("\n== Überdeckung (nominal 95 %)")
    for c in r["coverage"]:
        for row in c["rows"]:
            print(f"Fenster {c['length']:2d} n={c['n']:3d} t={row['t']:2d}  wahres S {row['true_s']:.3f}  log-log {row['loglog']:5.1f} % (undef {row['loglog_undef']})  "
                  f"einfach {row['plain']:5.1f} % (undef {row['plain_undef']})")
    print("\n== Mittlere Lebenszeit bis τ")
    for c in r["rmst"]:
        print(f"Fenster {c['length']} τ={c['tau']}: wahr {c['true']:.3f}  KM {fmt(c['km'])}  naiv Mittel von min(Zeit, τ) {fmt(c['naive_min_obs'])}")
