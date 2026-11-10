"""Die drei Wege (kpm_algorithm) und die Handrechnung mit Brüchen (kpm_evaluation.hand_rows) an der Mini-Instanz aus tests/conftest.py (DEMO-PLAYBOOK §1b)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import kpm_algorithm as A
import kpm_core as K
import kpm_evaluation as E
import kpm_scenario as S


def _sample(times, events) -> S.Sample:
    return S.Sample(np.asarray(times, dtype=float), np.asarray(events, dtype=bool))


def test_hand_rows_sind_die_handrechnung_mit_bruechen(six):
    rows = E.hand_rows(*six)
    assert [(r["t"], r["n"], r["d"]) for r in rows] == [(2, 6, 1), (4, 4, 1), (5, 3, 1), (8, 1, 1)]
    assert [r["factor"] for r in rows] == [F(5, 6), F(3, 4), F(2, 3), F(0)]
    assert [r["s"] for r in rows] == [F(5, 6), F(5, 8), F(5, 12), F(0)]


def test_hand_summary_ist_die_handrechnung(six):
    h = E.hand_summary(*six)
    assert h["median"] == 5 and h["mean_all"] == F(9, 2) and h["mean_events"] == F(19, 4)
    assert h["rmst"] == F(133, 24) and h["tau"] == 8 and h["greenwood_var"] == F(85, 1728)


def test_bruchrechnung_und_kern_stimmen_beim_handbeispiel_ueberein(six):
    km = K.kaplan_meier(*six)
    assert [float(r["s"]) for r in E.hand_rows(*six)] == pytest.approx(km["s"].tolist(), abs=1e-15)


def test_weg_km_beim_handbeispiel(six):
    w = A.way_km(_sample(*six))
    assert w.median == 5.0 and w.status == "ok" and math.isnan(w.mean)
    assert w.km["s"].tolist() == pytest.approx([5 / 6, 5 / 8, 5 / 12, 0.0])


def test_weg_zensierte_als_ereignis_ist_mittel_und_median_aller_zeiten(six):
    w = A.way_all(_sample(*six))
    assert w.mean == 4.5                       # (2 + 3 + 4 + 5 + 5 + 8) / 6
    assert w.median == 4.0                     # Kurve auf allen sechs Zeiten: S(4) = 3/6 = 0,5
    assert w.km["n"].tolist() == [6, 5, 4, 3, 1]   # t = 2, 3, 4, 5, 8 (bei 5 zwei Ereignisse, drei Container mit Zeit >= 5)
    assert w.km["s"][-1] == 0.0


def test_weg_nur_ereignisse_beim_handbeispiel(six):
    w = A.way_events(_sample(*six))
    assert w.mean == 4.75                      # (2 + 4 + 5 + 8) / 4
    assert w.median == 4.0                     # Zeiten 2, 4, 5, 8: S(4) = 2/4 = 0,5
    assert w.km["n"].tolist() == [4, 3, 2, 1]


def test_zensierte_als_ereignis_hat_dieselbe_kurve_wie_die_empirische_verteilung():
    rng = K.SplitMix64(4)
    obs = np.round(rng.lognormal(60, 4.0, 0.9), 1)    # mit Gleichständen
    w = A.way_all(_sample(obs, rng.uniform(60) < 0.5))
    for x in (1.0, 2.5, 4.0, 8.0):
        assert K.survival_at(w.km, x) == pytest.approx(float(np.mean(obs > x)), abs=1e-12)
    assert w.mean == pytest.approx(float(np.mean(obs)), abs=1e-12)


def test_status_nicht_schaetzbar_und_keine_ereignisse():
    s1 = _sample([1, 2, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0])
    w = A.way_km(s1)
    assert w.status == "nicht_schaetzbar" and math.isnan(w.median)
    we = A.way_events(s1)       # ein einziges Ereignis bei t = 1 mit n = 1: die Kurve fällt auf 0
    assert we.status == "ok" and we.median == 1.0 and we.km["s"].tolist() == [0.0]
    s0 = _sample([1, 2, 3], [0, 0, 0])
    for way in (A.way_km(s0), A.way_events(s0)):
        assert way.status == "keine_ereignisse" and math.isnan(way.median) and way.km["t"].size == 0
    assert A.way_all(s0).status == "ok" and A.way_all(s0).mean == 2.0   # als Ereignis gelesen gibt es immer eine Kurve


def test_restricted_mean_checked_status(six):
    s = _sample(*six)
    v, st = A.restricted_mean_checked(s, 4.0)
    assert st == "ok" and v == pytest.approx(11 / 3, abs=1e-12)
    v, st = A.restricted_mean_checked(s, 8.0)       # genau die größte Beobachtungszeit: erlaubt
    assert st == "ok" and v == pytest.approx(133 / 24, abs=1e-12)
    v, st = A.restricted_mean_checked(s, 8.5)
    assert st == "tau_ueber_beobachtung" and math.isnan(v)
    v, st = A.restricted_mean_checked(_sample([1, 2], [0, 0]), 1.0)
    assert st == "keine_ereignisse" and math.isnan(v)


def test_jede_einheit_ist_abschaltbar_die_wege_unterscheiden_sich():
    """Zweig-Test: auf einer Fenster-Stichprobe weichen die drei Wege voneinander ab (kein toter Zweig)."""
    s = S.simulate("Beobachtungsfenster (Ankunft zufällig)", 8, 300, 11)
    w = A.all_ways(s)
    medians = {n: w[n].median for n in A.WAYS}
    assert len({round(m, 6) for m in medians.values()}) == 3
    assert w[A.WAY_ALL].mean != w[A.WAY_EVENTS].mean


def test_quantile_rows_und_lowest_survival():
    a = E.analyse(E.Settings(300, 4, "Beobachtungsfenster (Ankunft zufällig)", 1))   # Median nicht schätzbar (tests/test_presets.py)
    rows = E.quantile_rows(a)
    assert [r["q"] for r in rows] == [0.1, 0.25, 0.5]
    assert math.isnan(rows[2]["estimate"]) and not math.isnan(rows[0]["estimate"])
    assert E.lowest_survival(a.km.km) > 0.5
