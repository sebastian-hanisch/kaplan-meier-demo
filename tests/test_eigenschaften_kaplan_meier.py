"""Eigenschaftstests (metamorph, ORACLE-PLAYBOOK §5.2): Beziehungen zwischen Ein- und Ausgabe, die gelten müssen, ohne dass man das richtige Ergebnis kennt.
Je Eigenschaft 30 bis 40 Zufallsinstanzen mit festem Seed, dazu Randfälle."""

import numpy as np
import pytest

import kpm_core as K


def _instance(rng, n):
    t = np.round(rng.lognormal(np.log(4.0), 0.9, n) * 2) / 2 + 0.5   # Raster: Gleichstände
    e = rng.uniform(size=n) < 0.6
    return t, e


def _instances(count=40, seed=1):
    rng = np.random.default_rng(seed)
    for k in range(count):
        t, e = _instance(rng, 5 + k)
        if e.any():
            yield t, e


def test_kurve_faellt_monoton_und_bleibt_in_null_eins():
    for t, e in _instances():
        km = K.kaplan_meier(t, e)
        assert np.all(np.diff(km["s"]) <= 1e-15) and km["s"].max() <= 1.0 and km["s"].min() >= 0.0
        assert np.all(km["n"] >= km["d"]) and np.all(np.diff(km["t"]) > 0)


def test_umsortieren_der_container_aendert_nichts():
    rng = np.random.default_rng(2)
    for t, e in _instances(30, 3):
        perm = rng.permutation(t.size)
        a, b = K.kaplan_meier(t, e), K.kaplan_meier(t[perm], e[perm])
        assert a["t"].tolist() == b["t"].tolist() and a["s"].tolist() == pytest.approx(b["s"].tolist(), abs=1e-15)


def test_skalierung_der_zeiten_skaliert_die_treppe_und_laesst_s_gleich():
    for t, e in _instances(30, 4):
        a, b = K.kaplan_meier(t, e), K.kaplan_meier(3.0 * t, e)
        assert b["t"].tolist() == pytest.approx((3.0 * a["t"]).tolist()) and b["s"].tolist() == pytest.approx(a["s"].tolist(), abs=1e-15)
        assert K.restricted_mean(b, 3.0 * 5.0) == pytest.approx(3.0 * K.restricted_mean(a, 5.0), rel=1e-12)   # Fläche skaliert mit


def test_verschiebung_aller_zeiten_um_c_verschiebt_die_treppe():
    for t, e in _instances(30, 5):
        a, b = K.kaplan_meier(t, e), K.kaplan_meier(t + 2.5, e)
        assert b["t"].tolist() == pytest.approx((a["t"] + 2.5).tolist()) and b["s"].tolist() == pytest.approx(a["s"].tolist(), abs=1e-15)


def test_zensierung_hinter_dem_letzten_ereignis_laesst_die_ereigniszeiten_und_hebt_die_werte():
    """Ein Zensierter hinter der letzten Ereigniszeit bleibt in jeder Risikomenge: die Stufen liegen an denselben Zeiten, aber jeder Faktor wird (n + 1 − d)/(n + 1).
    (Die Kurve bleibt also NICHT gleich: Die Annahme „ändert nichts“ aus dem Plan gilt nur für die Stufenzeiten, nicht für die Werte; README, Befunde gegenüber dem Plan.)"""
    for t, e in _instances(30, 6):
        last = t[e].max()
        keep = t <= last
        a = K.kaplan_meier(t[keep], e[keep])
        b = K.kaplan_meier(np.concatenate([t[keep], [last + 1.0]]), np.concatenate([e[keep], [False]]))
        assert a["t"].tolist() == b["t"].tolist()
        assert b["s"].tolist() == pytest.approx(np.cumprod(1.0 - a["d"] / (a["n"] + 1.0)).tolist(), abs=1e-15)
        assert np.all(b["s"] >= a["s"] - 1e-15) and b["s"][-1] > 0.0


def test_zensierung_vor_dem_ersten_ereignis_wirkt_wie_weglassen():
    for t, e in _instances(30, 7):
        first = t[e].min()
        early = first / 2.0
        a = K.kaplan_meier(t, e)
        b = K.kaplan_meier(np.concatenate([t, [early]]), np.concatenate([e, [False]]))
        assert a["t"].tolist() == b["t"].tolist() and a["s"].tolist() == pytest.approx(b["s"].tolist(), abs=1e-15)


def test_aus_einem_ereignis_eine_zensierung_machen_senkt_die_kurve_nie():
    """Wird ein Ereignis nachträglich zu einer Zensierung desselben Zeitpunkts, liegt die Kurve an keiner Stelle darunter (der Abgang fehlt)."""
    for t, e in _instances(30, 8):
        if e.sum() < 2:
            continue
        i = int(np.flatnonzero(e)[0])
        e2 = e.copy()
        e2[i] = False
        a, b = K.kaplan_meier(t, e), K.kaplan_meier(t, e2)
        for x in b["t"]:
            assert K.survival_at(b, float(x)) >= K.survival_at(a, float(x)) - 1e-12


def test_ohne_zensierung_ist_die_kurve_eins_minus_empirische_verteilungsfunktion():
    for t, _ in _instances(30, 9):
        km = K.kaplan_meier(t, np.ones(t.size, bool))
        for x in np.unique(t):
            assert K.survival_at(km, float(x)) == pytest.approx(float(np.mean(t > x)), abs=1e-12)


def test_band_wird_mit_dem_niveau_breiter_und_liegt_in_null_eins():
    for t, e in _instances(20, 10):
        km = K.kaplan_meier(t, e)
        for method in ("loglog", "plain"):
            lo90, hi90 = K.band_curve(km, 0.90, method)
            lo99, hi99 = K.band_curve(km, 0.99, method)
            ok = ~np.isnan(lo90)
            assert np.all(lo99[ok] <= lo90[ok] + 1e-12) and np.all(hi99[ok] >= hi90[ok] - 1e-12)
            assert np.all(lo99[ok] >= 0) and np.all(hi99[ok] <= 1)
            assert np.all(lo90[ok] <= km["s"][ok] + 1e-12) and np.all(hi90[ok] >= km["s"][ok] - 1e-12)


def test_mehr_container_verengen_das_band():
    """Dieselbe Welt, vierfaches n: die mittlere Bandbreite bei t = 4 sinkt (Mittel über 20 feste Stichproben, Faktor etwa 1/2 bei 4n)."""
    widths = {}
    for n in (100, 400):
        w = []
        for seed in range(20):
            rng = np.random.default_rng(300 + seed)
            d = rng.lognormal(np.log(4.0), 0.9, n)
            c = rng.uniform(0, 20, n)
            km = K.kaplan_meier(np.minimum(d, c), d <= c)
            _, lo, hi = K.greenwood_band(km, 4.0)
            w.append(hi - lo)
        widths[n] = float(np.mean(w))
    assert widths[400] < widths[100]
    assert widths[400] / widths[100] == pytest.approx(0.5, abs=0.08)
