"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstände von Ereignis und Zensierung, nichts zensiert, alles zensiert,
Zensierung vor dem ersten Ereignis, Status statt stiller Zahlen."""

import math

import numpy as np
import pytest

import kpm_algorithm as A
import kpm_constants as C
import kpm_core as K
import kpm_evaluation as E
import kpm_scenario as S


def _s(times, events):
    return S.Sample(np.asarray(times, dtype=float), np.asarray(events, dtype=bool))


def test_ein_container_mit_ereignis():
    km = K.kaplan_meier([3.0], [1])
    assert km["t"].tolist() == [3.0] and km["s"].tolist() == [0.0] and K.quantile(km, 0.5) == 3.0
    assert math.isnan(K.greenwood_band(km, 3.0)[1])              # S = 0: kein Band, kein Fehler
    assert A.all_ways(_s([3.0], [1]))[A.WAY_KM].median == 3.0


def test_ein_container_zensiert_hat_keine_kurve_und_keinen_median():
    km = K.kaplan_meier([3.0], [0])
    assert km["t"].size == 0
    w = A.all_ways(_s([3.0], [0]))
    assert w[A.WAY_KM].status == "keine_ereignisse" and w[A.WAY_EVENTS].status == "keine_ereignisse"
    assert w[A.WAY_ALL].median == 3.0 and w[A.WAY_ALL].mean == 3.0        # als Ereignis gelesen: eine Abholung bei 3


def test_genau_zwei_container():
    km = K.kaplan_meier([2.0, 5.0], [1, 1])
    assert km["s"].tolist() == [0.5, 0.0] and K.quantile(km, 0.5) == 2.0
    km2 = K.kaplan_meier([2.0, 5.0], [1, 0])
    assert km2["s"].tolist() == [0.5] and K.quantile(km2, 0.5) == 2.0
    km3 = K.kaplan_meier([2.0, 5.0], [0, 1])          # Zensierung vor dem ersten Ereignis: der zweite ist allein im Risiko
    assert km3["n"].tolist() == [1.0] and km3["s"].tolist() == [0.0] and K.quantile(km3, 0.5) == 5.0


def test_gleichstand_ereignis_und_zensierung_bei_n_gleich_zwei():
    km = K.kaplan_meier([5.0, 5.0], [1, 0])
    assert km["n"].tolist() == [2.0] and km["s"].tolist() == [0.5]       # der Zensierte bleibt im Risiko


def test_alle_container_haben_dieselbe_zeit():
    km = K.kaplan_meier([4.0] * 5, [1] * 5)
    assert km["t"].tolist() == [4.0] and km["s"].tolist() == [0.0] and km["gw"].tolist() == [0.0]
    km = K.kaplan_meier([4.0] * 5, [1, 1, 0, 0, 0])
    assert km["s"].tolist() == pytest.approx([0.6]) and km["n"].tolist() == [5.0]


def test_nichts_zensiert_alle_drei_wege_stimmen_ueberein():
    s = S.simulate(C.KIND_FIXED, 60, 50, 3)
    assert s.n_events >= s.n - 1       # Frist 60 Tage: höchstens ein Container ist beim Ende noch da (für diesen Zufalls-Seed)
    ev = _s(s.obs[s.event], np.ones(s.n_events, bool))
    w = A.all_ways(ev)
    assert w[A.WAY_ALL].median == w[A.WAY_EVENTS].median == w[A.WAY_KM].median
    assert w[A.WAY_ALL].km["s"].tolist() == w[A.WAY_KM].km["s"].tolist()


def test_alle_zensiert_meldet_status_statt_stiller_zahlen():
    a = E.analyse(E.Settings(20, 2, C.KIND_FIXED, 1))
    s0 = _s([2.0] * 20, [0] * 20)
    for w in (A.way_km(s0), A.way_events(s0)):
        assert w.status == "keine_ereignisse" and math.isnan(w.median) and math.isnan(w.mean)
    assert A.restricted_mean_checked(s0, 1.0)[1] == "keine_ereignisse"
    assert E.verdict_core(E.Analysis(a.settings, S.Sample(s0.obs, s0.event), A.all_ways(s0), 1.0, 4.0, S.true_mean()))[0] == "warning"
    assert E.band_checks(E.Analysis(a.settings, s0, A.all_ways(s0), 1.0, 4.0, S.true_mean()), "loglog") == []


def test_kleinste_und_groesste_einstellungen_laufen_ohne_fehler():
    for kind in C.KINDS:
        for window in (C.W_MIN, C.W_MAX):
            for n in (C.N_MIN, C.N_MAX):
                a = E.analyse(E.Settings(n, window, kind, 1))
                assert a.sample.n == n
                E.verdict_core(a)
                E.band_checks(a, "plain")
                E.quantile_rows(a)


def test_tau_genau_an_der_groessten_beobachtungszeit_und_darueber():
    s = _s([1.0, 2.0, 3.0, 6.0], [1, 0, 1, 0])
    assert A.restricted_mean_checked(s, 6.0)[1] == "ok"
    assert A.restricted_mean_checked(s, 6.000001)[1] == "tau_ueber_beobachtung"


def test_eingabefehler_des_kerns_sind_verstaendlich():
    with pytest.raises(K.SurvivalError, match="größer als 0"):
        K.kaplan_meier([0.0, 1.0], [1, 1])
    with pytest.raises(K.SurvivalError, match="0 \\(zensiert\\) oder 1"):
        K.kaplan_meier([1.0, 2.0], [1, 3])
    with pytest.raises(K.SurvivalError, match="mindestens eine Beobachtung"):
        K.kaplan_meier([], [])


def test_worst_case_am_slider_maximum_laeuft_durch():
    """Größtes n mit den 30 festen Zufalls-Seeds für beide Zensierungsarten (das rechnet die App bei jeder Änderung): Hänger und Aufblähen fangen, keine Zeit-Assertion (DEMO-PLAYBOOK §1c)."""
    for kind in C.KINDS:
        rep = E.repeat_summary(kind, C.W_MAX, C.N_MAX)
        assert rep["seeds"] == len(C.BAND_SEEDS) == 30
        assert all(len(rep["ways"][name]["median"]) == 30 for name in A.WAYS)
