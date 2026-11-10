"""Aussagen der README: jede Zahl steht in der eingefrorenen Messung (tools/messung_kaplan_meier.json), die Messung stimmt mit einer Neuberechnung überein,
und die Zahlen der README stimmen mit der Messung überein (Drift-Wächter, Voraussetzung im selben Satz wie in der README).

Die Messdatei wird von tools/measure_kaplan_meier.py erzeugt (SplitMix64, feste Zufalls-Seeds). Hier wird sie nicht überschrieben."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import kpm_constants as C  # noqa: E402
import kpm_evaluation as E  # noqa: E402
import measure_kaplan_meier as MK  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")
M = E.measurements()
TRUE_MEAN = M["meta"]["mean_true"]
de, pct = E.de, E.pct


def _cell(d):
    return f"{de(d['mean'])} ± {de(d['se'])}"


def _find(block, kind, length, n=300):
    return next(c for c in M[block] if c["kind"] == kind and c["length"] == length and c["n"] == n)


def _dev_mean(c, key="mean_all"):
    return pct(E.deviation_percent(c[key]["mean"], TRUE_MEAN))


def _dev_med(c, key):
    return pct(E.deviation_percent(c[key]["mean"], C.MEDIAN_TRUE))


# ------------------------------------------------------------------------------------------------ Messdatei gegen Neuberechnung


def test_messdatei_meta_und_wahrheit():
    assert M["meta"]["runs"] == 1000 and M["meta"]["cover_runs"] == 1000
    assert M["meta"]["median_true"] == 4.0 and M["meta"]["sigma"] == 0.9
    assert M["meta"]["mean_true"] == pytest.approx(5.99721, abs=1e-5) and M["meta"]["q90_true"] == pytest.approx(12.675751, abs=1e-5)


@pytest.mark.parametrize("block,kind,length,n", [("cells", C.KIND_WINDOW, 30, 300), ("cells", C.KIND_WINDOW, 4, 300), ("cells", C.KIND_FIXED, 8, 300), ("v1", C.KIND_WINDOW, 30, 100)])
def test_messung_gegen_neu_berechnung(block, kind, length, n):
    """Die Zelle wird mit denselben Zufalls-Seeds neu gerechnet; Mittel auf 0,005 Tage, Anteile auf 0,2 Prozentpunkte (Plattformunterschiede in der letzten Stelle)."""
    old = _find(block, kind, length, n)
    new = MK.cell(kind, length, n, old["base"], old["runs"])
    for key in ("mean_all", "mean_events", "median_all", "median_events", "median_km"):
        assert new[key]["mean"] == pytest.approx(old[key]["mean"], abs=5e-3), (key, new[key], old[key])
        assert new[key]["defined"] == pytest.approx(old[key]["defined"], abs=2)
    assert new["censored"]["mean"] == pytest.approx(old["censored"]["mean"], abs=2e-3)
    assert new["undef_km"] == pytest.approx(old["undef_km"], abs=2e-3)


def test_ueberdeckung_neu_berechnet():
    old = M["coverage"][0]                    # Fenster 30, n = 100, t = 2, 4, 8, 12
    new = MK.coverage(old["length"], old["n"], [r["t"] for r in old["rows"]], old["base"], old["runs"])
    for o, nw in zip(old["rows"], new["rows"]):
        assert nw["loglog"] == pytest.approx(o["loglog"], abs=0.2) and nw["plain"] == pytest.approx(o["plain"], abs=0.2)
        assert nw["loglog_undef"] == o["loglog_undef"]


def test_mittlere_lebenszeit_neu_berechnet():
    old = M["rmst"][0]
    new = MK.rmst_cell(old["length"], old["tau"], old["n"], old["base"], old["runs"])
    assert new["true"] == pytest.approx(old["true"], abs=1e-9)
    assert new["km"]["mean"] == pytest.approx(old["km"]["mean"], abs=5e-3) and new["naive_min_obs"]["mean"] == pytest.approx(old["naive_min_obs"]["mean"], abs=5e-3)


def test_messdatei_innere_zusammenhaenge():
    """Gleiche Seeds, gleiche Zellen: die Zeilen der Messdatei, die dieselbe Bedingung beschreiben, sind gleich."""
    assert _find("defined", C.KIND_WINDOW, 30) == _find("cells", C.KIND_WINDOW, 30) == _find("v1", C.KIND_WINDOW, 30)
    assert _find("defined", C.KIND_WINDOW, 4) == _find("cells", C.KIND_WINDOW, 4) == _find("defined_n", C.KIND_WINDOW, 4)
    assert _find("defined", C.KIND_WINDOW, 8) == _find("cells", C.KIND_WINDOW, 8)


def test_ereignisanteile_passen_zur_geschlossenen_formel_der_zensierung():
    """Feste Frist c: zensiert = S(c) (Mittel über 1000 Läufe mit n = 300 hat einen Standardfehler von unter 0,1 Prozentpunkten)."""
    import kpm_scenario as S

    for length in (30, 8, 4):
        c = _find("cells", C.KIND_FIXED, length)
        assert c["censored"]["mean"] == pytest.approx(float(S.true_survival(length)), abs=0.003)


# ------------------------------------------------------------------------------------------------ README gegen Messdatei


def test_readme_zahlen():
    w30, w8, w4 = (_find("cells", C.KIND_WINDOW, x) for x in (30, 8, 4))
    f30, f8, f4 = (_find("cells", C.KIND_FIXED, x) for x in (30, 8, 4))
    v100, v300, v1000 = M["v1"]
    n100, n300, n1000 = M["defined_n"]
    needed = [
        # Einleitung „Warum dieses Problem“
        f"{de(100 * w30['censored']['mean'], 1)} % zensiert, 1000 Läufe",
        f"liegt bei {_cell(w30['mean_all'])} Tagen gegen das wahre Mittel {de(TRUE_MEAN)} ({_dev_mean(w30)} gegen die Wahrheit)",
        f"der Median von Kaplan–Meier bei {_cell(w30['median_km'])} gegen den wahren Median {de(C.MEDIAN_TRUE)}",
        f"Fenster von 8 Tagen ({de(100 * w8['censored']['mean'], 1)} % zensiert) sinkt das Mittel auf {de(w8['mean_all']['mean'])} ({_dev_mean(w8)})",
        f"Fenster von 4 Tagen (= wahrer Median, n = 300, {de(100 * w4['censored']['mean'], 1)} % zensiert) hat Kaplan–Meier in {de(100 * w4['undef_km'], 1)} % der Läufe keinen Median",
        # Korrekturen gegenüber dem Plan
        f"{_cell(w30['median_all'])} („Zensierte als Ereignis“) und {_cell(w30['median_events'])} („nur Ereignisse“) bei Fenster 30 Tage, n = 300 ({_dev_med(w30, 'median_all')} und {_dev_med(w30, 'median_events')} gegen den wahren Median)",
        f"hier {de(w30['mean_all']['mean'])}, {de(w30['median_km']['mean'])} und {de(100 * w4['undef_km'], 1)} %",
        f"{de(100 * (1 - n100['undef_km']), 1)} % (n = 100), {de(100 * (1 - n300['undef_km']), 1)} % (n = 300), {de(100 * (1 - n1000['undef_km']), 1)} % (n = 1000) der Läufe mit Median",
        f"({de(n100['median_km']['mean'])}, {de(n300['median_km']['mean'])}, {de(n1000['median_km']['mean'])})",
        f"{de(E.coverage_range('loglog')[0], 1)} bis {de(E.coverage_range('loglog')[1], 1)} % (log-log) und {de(E.coverage_range('plain')[0], 1)} bis {de(E.coverage_range('plain')[1], 1)} % (einfach)",
        f"({_cell(f30['median_all'])} bei Frist 30 und 8 Tage, wahr {de(C.MEDIAN_TRUE)}; untere Median-Konvention, beide Fristen nutzen dieselben Zufalls-Seeds und sind keine zwei unabhängigen Bestätigungen), das Mittel nicht ({de(f30['mean_all']['mean'])} und {de(f8['mean_all']['mean'])}, wahr {de(TRUE_MEAN)})",
        # Tabelle der Befunde
        f"{_cell(w30['mean_all'])} ({_dev_mean(w30)}), {_cell(w30['mean_events'])} ({_dev_mean(w30, 'mean_events')}), {_cell(w30['median_km'])} (wahr: Mittel {de(TRUE_MEAN)}, Median {de(C.MEDIAN_TRUE)})",
        f"{_cell(w8['mean_all'])} ({_dev_mean(w8)}), {_cell(w8['mean_events'])} ({_dev_mean(w8, 'mean_events')}), {_cell(w8['median_km'])}",
        f"{_cell(w4['mean_all'])} ({_dev_mean(w4)}); Median in {de(100 * (1 - w4['undef_km']), 1)} % der Läufe vorhanden, dort {_cell(w4['median_km'])}",
        f"{_cell(v100['mean_all'])}, {_cell(v300['mean_all'])}, {_cell(v1000['mean_all'])}: mehr Daten reparieren das Mittel nicht ({_dev_mean(v100)}, {_dev_mean(v300)}, {_dev_mean(v1000)})",
        f"„Zensierte als Ereignis“ {de(w30['median_all']['mean'])}, {de(w8['median_all']['mean'])}, {de(w4['median_all']['mean'])}; „nur Ereignisse“ {de(w30['median_events']['mean'])}, {de(w8['median_events']['mean'])}, {de(w4['median_events']['mean'])}",
        f"({de(100 * f30['censored']['mean'], 1)} %, {de(100 * f8['censored']['mean'], 1)} %, {de(100 * f4['censored']['mean'], 1)} % zensiert)",
        f"{_cell(f30['median_all'])}, {_cell(f8['median_all'])}, {_cell(f4['median_all'])}; Median „nur Ereignisse“ {de(f30['median_events']['mean'])}, {de(f8['median_events']['mean'])}, {de(f4['median_events']['mean'])}; "
        f"Mittel aller {de(f30['mean_all']['mean'])}, {de(f8['mean_all']['mean'])}, {de(f4['mean_all']['mean'])}",
        ", ".join(de(100 * (1 - _find('defined', C.KIND_WINDOW, x)["undef_km"]), 1) + " %" for x in (4, 5, 6, 8, 12, 30)) + " der Läufe",
        f"log-log {de(E.coverage_range('loglog')[0], 1)} bis {de(E.coverage_range('loglog')[1], 1)} %; einfach {de(E.coverage_range('plain')[0], 1)} bis {de(E.coverage_range('plain')[1], 1)} %",
    ]
    for r in M["rmst"]:
        needed.append(f"(wahr {de(r['true'], 3)}) | Kaplan–Meier {_cell(r['km'])} ({pct(E.deviation_percent(r['km']['mean'], r['true']))}); "
                      f"Mittel von min(Zeit, τ) {_cell(r['naive_min_obs'])} ({pct(E.deviation_percent(r['naive_min_obs']['mean'], r['true']))})")
    missing = [s for s in needed if s not in README]
    assert not missing, missing


def test_readme_nennt_die_zahl_der_tests_der_suite():
    import re
    import subprocess

    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", str(ROOT / "tests"), "-p", "no:cacheprovider"], capture_output=True, text=True, cwd=ROOT, timeout=300).stdout
    m = re.search(r"(\d+) tests? collected", out)
    assert m, out[-300:]
    assert f"{m.group(1)} Tests (`pytest tests/`" in README


def test_ueberdeckungsspannen_der_readme_gelten_fuer_beide_bandarten_und_alle_elf_zellen():
    cells = [(b["length"], b["n"], r["t"]) for b in M["coverage"] for r in b["rows"]]
    assert len(cells) == 11 and "elf Zellen" in README
    assert all(b["runs"] == 1000 for b in M["coverage"])
    ll = [r["loglog"] for b in M["coverage"] for r in b["rows"]]
    pl = [r["plain"] for b in M["coverage"] for r in b["rows"]]
    assert min(ll) > min(pl)                                  # das log-log-Band liegt im ungünstigsten Fall näher an 95 %
    worst = min((r["plain"], b["length"], b["n"], r["t"]) for b in M["coverage"] for r in b["rows"])
    assert worst[1:] == (30, 100, 12)                         # „am kleinsten bei Fenster 30 Tage, n = 100, t = 12“
    assert sum(r["loglog_undef"] for b in M["coverage"] for r in b["rows"] if (b["n"], r["t"]) == (100, 12)) == 2   # „zwei der 1000 Läufe haben dort kein Band“


def test_fixe_aussagen_der_readme_ueber_die_naiven_wege():
    """Rangfolge: bei jedem Fenster liegt der Median von Kaplan–Meier näher an 4 als beide naiven Mediane, im Mittel über die Läufe (Fenster 30, 8, 4 Tage)."""
    for length in (30, 8, 4):
        c = _find("cells", C.KIND_WINDOW, length)
        km = abs(c["median_km"]["mean"] - 4.0)
        assert km < abs(c["median_all"]["mean"] - 4.0) and km < abs(c["median_events"]["mean"] - 4.0), length
    # Bei der festen Frist ist „Zensierte als Ereignis“ beim Median so gut wie Kaplan–Meier, wenn die Frist über dem wahren Median liegt, „nur Ereignisse“ nicht
    for length in (30, 8):
        c = _find("cells", C.KIND_FIXED, length)
        assert abs(c["median_all"]["mean"] - c["median_km"]["mean"]) < 0.02 and abs(c["median_events"]["mean"] - 4.0) > 0.05


def test_standardfehler_der_schaetzbaren_anteile_ist_rund_1_6_prozentpunkte():
    """„Binomial-Standardfehler je Wert rund 1,6 Prozentpunkte“ (App und README) bei Fenster 4 Tage, 1000 Läufe je Zelle."""
    import math

    for c in M["defined_n"]:
        p = 1.0 - c["undef_km"]
        assert 100 * math.sqrt(p * (1 - p) / c["runs"]) == pytest.approx(1.6, abs=0.1)
    spread = max(1 - c["undef_km"] for c in M["defined_n"]) - min(1 - c["undef_km"] for c in M["defined_n"])
    assert spread < 0.04       # alle drei Anteile liegen höchstens vier Prozentpunkte auseinander
