"""Szenario: die Erzeuger der Stichproben (Fenster, feste Frist) und die Wahrheit in geschlossener Form."""

import math

import numpy as np
import pytest

import kpm_constants as C
import kpm_scenario as S


def test_simulate_ist_deterministisch_und_seed_abhaengig():
    a = S.simulate(C.KIND_WINDOW, 8, 100, 5)
    b = S.simulate(C.KIND_WINDOW, 8, 100, 5)
    c = S.simulate(C.KIND_WINDOW, 8, 100, 6)
    assert np.array_equal(a.obs, b.obs) and np.array_equal(a.event, b.event)
    assert not np.array_equal(a.obs, c.obs)


def test_fenster_beobachtungsdauer_liegt_in_null_bis_laenge_und_ereignis_heisst_abholung_gesehen():
    s = S.simulate(C.KIND_WINDOW, 8, 2000, 1)
    assert s.obs.min() > 0 and s.obs.max() <= 8.0
    assert 0 < s.n_events < s.n
    assert s.censored_share == pytest.approx(1 - s.n_events / s.n)


def test_feste_frist_zensiert_genau_bei_der_frist():
    s = S.simulate(C.KIND_FIXED, 8, 2000, 1)
    assert s.obs.max() == 8.0
    assert np.all(s.obs[~s.event] == 8.0)          # jeder Zensierte steht genau bei der Frist
    assert np.all(s.obs[s.event] <= 8.0)


def test_zensierungsanteil_passt_zur_geschlossenen_formel():
    """Feste Frist c: Anteil zensiert = S(c). Fenster W: Mittel über die Ankunft U(0, W) von S(u), u gleichverteilt in (0, W)."""
    for c in (4, 8, 30):
        shares = [S.simulate(C.KIND_FIXED, c, 2000, 100 + i).censored_share for i in range(10)]
        assert np.mean(shares) == pytest.approx(float(S.true_survival(c)), abs=0.01), c
    w = 8
    grid = (np.arange(4000) + 0.5) * w / 4000
    want = float(np.mean(S.true_survival(grid)))
    shares = [S.simulate(C.KIND_WINDOW, w, 2000, 200 + i).censored_share for i in range(10)]
    assert np.mean(shares) == pytest.approx(want, abs=0.01)


def test_ungueltige_eingaben_werden_abgelehnt():
    with pytest.raises(ValueError):
        S.simulate("irgendwas", 8, 10, 1)
    with pytest.raises(ValueError):
        S.simulate(C.KIND_WINDOW, 0, 10, 1)
    with pytest.raises(ValueError):
        S.simulate(C.KIND_WINDOW, 8, 0, 1)


def test_wahrheit_median_quantil_und_mittel():
    assert float(S.true_survival(4.0)) == pytest.approx(0.5, abs=1e-12)
    assert float(S.true_survival(0.0)) == 1.0 and float(S.true_survival(-1.0)) == 1.0
    assert S.true_quantile(0.5) == pytest.approx(4.0, abs=1e-12)
    for q in (0.1, 0.25, 0.9):
        assert float(S.true_survival(S.true_quantile(q))) == pytest.approx(1 - q, abs=1e-9)
    assert S.true_mean() == pytest.approx(4 * math.exp(0.405), rel=1e-12) and round(S.true_mean(), 2) == 6.00
    assert S.true_quantile(0.9) == pytest.approx(12.676, abs=1e-3)


def test_wahre_mittlere_lebenszeit_gegen_gitterquadratur():
    for tau in (1.0, 6.0, 8.0, 40.0):
        grid = (np.arange(200000) + 0.5) * tau / 200000
        want = float(np.mean(S.true_survival(grid))) * tau
        assert S.true_restricted_mean(tau) == pytest.approx(want, rel=1e-6), tau
    assert S.true_restricted_mean(0.0) == 0.0
    assert S.true_restricted_mean(1e6) == pytest.approx(S.true_mean(), rel=1e-6)   # τ → ∞ gibt das Mittel
