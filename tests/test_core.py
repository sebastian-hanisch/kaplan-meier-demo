"""Tests des kopierten Kerns kpm_core (aus ueberlebenszeit-planung/kern/test_uz_core.py, auf die hier genutzten Funktionen gekürzt)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import kpm_core as K

SIX_T = [2, 3, 4, 5, 5, 8]
SIX_E = [1, 0, 1, 1, 0, 1]


def test_km_handbeispiel_sechs_container():
    km = K.kaplan_meier(SIX_T, SIX_E)
    assert km["t"].tolist() == [2, 4, 5, 8]
    assert km["n"].tolist() == [6, 4, 3, 1]
    assert km["d"].tolist() == [1, 1, 1, 1]
    for got, want in zip(km["s"], [F(5, 6), F(5, 8), F(5, 12), F(0)]):
        assert got == pytest.approx(float(want), abs=1e-15)
    assert K.quantile(km, 0.5) == 5.0
    assert K.restricted_mean(km, 8.0) == pytest.approx(133 / 24, abs=1e-12)
    # Greenwood-Varianz bei t = 5: S² Σ d/(n(n−d)) = 85/1728
    assert km["s"][2] ** 2 * km["gw"][2] == pytest.approx(85 / 1728, abs=1e-12)


def test_mittel_aller_und_nur_ereignisse_beim_handbeispiel():
    assert float(np.mean(SIX_T)) == 4.5
    assert float(np.mean([t for t, e in zip(SIX_T, SIX_E) if e])) == 4.75


def test_gleichstand_ereignis_vor_zensierung():
    """Bei t = 5 stehen eine Abholung und eine Zensierung: der Zensierte bleibt im Risiko (n = 3, nicht 2)."""
    km = K.kaplan_meier(SIX_T, SIX_E)
    assert km["n"][2] == 3
    # Gegenprobe: wäre die Zensierung vor dem Ereignis abgezogen, wäre n = 2 und S(5) = 5/8 · 1/2 = 5/16
    assert km["s"][2] != pytest.approx(5 / 16)


def test_km_ohne_zensierung_ist_eins_minus_verteilungsfunktion():
    rng = K.SplitMix64(1)
    t = np.round(rng.uniform(50) * 8 + 0.5, 1)  # mit Gleichständen
    km = K.kaplan_meier(t, np.ones(50, bool))
    for x in (0.5, 1.0, 2.0, 4.0, 8.0):
        assert K.survival_at(km, x) == pytest.approx(float(np.mean(t > x)), abs=1e-12)


def test_zensierung_hinter_letztem_ereignis_laesst_die_stufenzeiten_gleich():
    a = K.kaplan_meier([2, 4, 6, 9, 11], [1, 1, 1, 0, 0])
    assert a["t"].tolist() == [2, 4, 6] and a["n"].tolist() == [5, 4, 3]   # die beiden Zensierten bleiben in jeder Risikomenge
    assert a["s"].tolist() == pytest.approx([1 - 1 / 5, (1 - 1 / 5) * (1 - 1 / 4), (1 - 1 / 5) * (1 - 1 / 4) * (1 - 1 / 3)])


def test_alle_zensiert_gibt_leere_kurve_und_keinen_median():
    km = K.kaplan_meier([1, 2, 3], [0, 0, 0])
    assert km["t"].size == 0 and math.isnan(K.quantile(km, 0.5)) and K.survival_at(km, 5.0) == 1.0


def test_quantil_nicht_schaetzbar_wenn_kurve_nicht_faellt():
    km = K.kaplan_meier([1, 2, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0])
    assert math.isnan(K.quantile(km, 0.5))
    assert K.quantile(km, 0.1) == 1.0   # S(1) = 5/6 <= 0,9


def test_greenwood_band_liegt_in_null_eins_und_um_die_kurve():
    km = K.kaplan_meier(SIX_T, SIX_E)
    for method in ("loglog", "plain"):
        s, lo, hi = K.greenwood_band(km, 4.5, 0.95, method)
        assert s == pytest.approx(5 / 8) and 0 <= lo < s < hi <= 1
    s, lo, hi = K.greenwood_band(km, 9.0)
    assert s == 0.0 and math.isnan(lo)
    s, lo, hi = K.greenwood_band(km, 1.0)  # vor der ersten Ereigniszeit
    assert s == 1.0 and math.isnan(lo)
    assert K.z_two_sided(0.95) == pytest.approx(1.959964, abs=1e-6)


def test_greenwood_band_einfach_hat_die_handformel():
    km = K.kaplan_meier(SIX_T, SIX_E)
    s, lo, hi = K.greenwood_band(km, 5.0, 0.95, "plain")
    se = math.sqrt(85 / 1728)
    assert (s, lo, hi) == pytest.approx((5 / 12, max(0.0, 5 / 12 - 1.959964 * se), min(1.0, 5 / 12 + 1.959964 * se)), abs=1e-5)


def test_band_curve_stimmt_mit_dem_punktweisen_band_ueberein():
    rng = K.SplitMix64(5)
    t = rng.lognormal(80, 4.0, 0.9)
    e = rng.uniform(80) < 0.7
    km = K.kaplan_meier(t, e)
    for method in ("loglog", "plain"):
        lo, hi = K.band_curve(km, 0.95, method)
        for i, x in enumerate(km["t"]):
            s, lo1, hi1 = K.greenwood_band(km, float(x), 0.95, method)
            if math.isnan(lo1):
                assert math.isnan(lo[i]) and math.isnan(hi[i])
            else:
                assert lo[i] == pytest.approx(lo1, abs=1e-12) and hi[i] == pytest.approx(hi1, abs=1e-12)


def test_unbekanntes_band_verfahren_und_niveau_melden_fehler():
    km = K.kaplan_meier(SIX_T, SIX_E)
    with pytest.raises(K.SurvivalError):
        K.greenwood_band(km, 4.0, 0.95, "exakt")
    with pytest.raises(K.SurvivalError):
        K.band_curve(km, 0.95, "exakt")
    with pytest.raises(K.SurvivalError):
        K.z_two_sided(1.0)


def test_restricted_mean_bis_tau_und_fehler():
    km = K.kaplan_meier(SIX_T, SIX_E)
    # bis 4: 2 · 1 + 2 · 5/6 = 11/3
    assert K.restricted_mean(km, 4.0) == pytest.approx(11 / 3, abs=1e-12)
    assert K.restricted_mean(km, 1.0) == pytest.approx(1.0)
    with pytest.raises(K.SurvivalError):
        K.restricted_mean(km, 0.0)


def test_eingabepruefung():
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([0, 1], [1, 1])  # Zeit 0
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([1, 2], [1, 2])  # Ereignis 2
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([1, 2], [1])  # Länge
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([], [])
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([1.0, float("nan")], [1, 1])


def test_splitmix_referenzwert_und_determinismus():
    # erster Ausgabewert von SplitMix64 bei Seed 0 (Referenz: Vigna, splitmix64.c)
    assert int(K.SplitMix64(0).u64(1)[0]) == 0xE220A8397B1DCDAF
    a, b = K.SplitMix64(5), K.SplitMix64(5)
    assert np.array_equal(a.uniform(10), b.uniform(10))
    u = K.SplitMix64(9).uniform(50000)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01
    z = K.SplitMix64(9).normal(50000)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02


def test_lognormal_hat_median_und_streuung():
    x = K.SplitMix64(3).lognormal(60000, 4.0, 0.9)
    assert np.median(x) == pytest.approx(4.0, rel=0.02)
    assert np.std(np.log(x)) == pytest.approx(0.9, rel=0.02)
