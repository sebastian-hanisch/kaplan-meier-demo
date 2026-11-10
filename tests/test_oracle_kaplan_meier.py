"""Orakel-Tests: Kaplan–Meier, Greenwood-Varianz, mittlere Lebenszeit und die Wahrheit der Welt gegen UNABHÄNGIGE Rechenwege (ORACLE-PLAYBOOK §2, DEMO-PLAYBOOK §12).

Unabhängig heißt: kein Aufruf von kpm_core im Orakel-Zweig.
* Kaplan–Meier gegen das Umverteilungsverfahren nach rechts (Efron 1967), in exakten Brüchen: jede Zensierung gibt ihre Masse zu gleichen Teilen an alle späteren Beobachtungen ab.
  Das ist ein anderer Rechenweg als das Produkt der Faktoren (n − d)/n.
* Ohne Zensierung gegen die Aufzählung: Anteil der Zeiten, die länger sind als t.
* Greenwood-Standardfehler gegen eine Bootstrap-Schätzung (numpy-Zufall, anderer Strom als SplitMix64), Band-Formeln gegen eine Handrechnung in Brüchen und Logarithmen.
* Mittlere Lebenszeit gegen die Fläche unter der Treppe in Brüchen und gegen eine feine Gitter-Quadratur.
* Die Wahrheit (Überlebensfunktion, Quantil, mittlere Lebenszeit) gegen scipy.stats.lognorm und scipy.integrate.quad.
* Unverzerrtheit von Kaplan–Meier an einer Welt, die mit numpy statt SplitMix64 erzeugt wird (Mittel über Läufe in einem Band um die Wahrheit).

Das Orakel wurde vorher am Handbeispiel (sechs Container: 5/6, 5/8, 5/12, 0) geprüft."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import kpm_core as K
import kpm_scenario as S


# ------------------------------------------------------------------------------------------------ Orakel: Umverteilung nach rechts


def redistribute_survival(times, events):
    """Efron: Masse 1/n je Beobachtung, sortiert nach (Zeit, Ereignis zuerst); jede Zensierung verteilt ihre Masse zu gleichen Teilen auf alle späteren Positionen.
    Gibt {Ereigniszeit: S(t)} mit S(t) = 1 − Summe der Massen der Ereignisse bis t."""
    n = len(times)
    order = sorted(range(n), key=lambda i: (times[i], 0 if events[i] else 1))
    mass = [F(1, n) for _ in order]
    for pos, i in enumerate(order):
        if not events[i]:
            later = list(range(pos + 1, n))
            if later:
                share = mass[pos] / len(later)
                for j in later:
                    mass[j] += share
            mass[pos] = F(0)
    out, cum = {}, F(0)
    for pos, i in enumerate(order):
        if events[i]:
            cum += mass[pos]
            out[times[i]] = 1 - cum
    return out


def random_instance(rng, n, grid=0.5):
    """Zeiten auf einem Raster (viele Gleichstände, auch zwischen Ereignissen und Zensierungen); Zensierungsanteil um 40 %."""
    t = [float(x) for x in np.round(rng.lognormal(np.log(4.0), 0.9, n) / grid) * grid]
    t = [x if x > 0 else grid for x in t]
    e = [bool(x) for x in (rng.uniform(size=n) < 0.6)]
    return t, e


def test_oracle_handbeispiel_umverteilung():
    s = redistribute_survival([2, 3, 4, 5, 5, 8], [1, 0, 1, 1, 0, 1])
    assert s == {2: F(5, 6), 4: F(5, 8), 5: F(5, 12), 8: F(0)}


def test_kaplan_meier_gleich_umverteilung_auf_zufallsinstanzen_mit_gleichstaenden():
    rng = np.random.default_rng(20261110)
    checked = 0
    for k in range(120):
        n = 2 + k % 40
        t, e = random_instance(rng, n)
        if not any(e):
            continue
        km = K.kaplan_meier(t, e)
        want = redistribute_survival(t, e)
        assert km["t"].tolist() == sorted(want), k
        assert km["s"].tolist() == pytest.approx([float(want[x]) for x in km["t"]], abs=1e-12), k
        checked += 1
    assert checked >= 100


def test_ohne_zensierung_gleich_aufzaehlung_der_laengeren_zeiten():
    rng = np.random.default_rng(7)
    for k in range(40):
        n = 3 + k
        t, _ = random_instance(rng, n)
        km = K.kaplan_meier(t, [True] * n)
        for x in km["t"]:
            assert K.survival_at(km, float(x)) == pytest.approx(sum(1 for v in t if v > x) / n, abs=1e-12)


# ------------------------------------------------------------------------------------------------ Orakel: Greenwood


def greenwood_by_hand(times, events, t0):
    """S(t0) und Varianz S² Σ d/(n(n−d)) in Brüchen, n und d durch Abzählen je Ereigniszeit."""
    s, acc = F(1), F(0)
    for x in sorted({v for v, e in zip(times, events) if e and v <= t0}):
        n = sum(1 for v in times if v >= x)
        d = sum(1 for v, e in zip(times, events) if v == x and e)
        s *= F(n - d, n)
        if n > d:
            acc += F(d, n * (n - d))
    return s, s * s * acc


def test_greenwood_gegen_bruchrechnung():
    rng = np.random.default_rng(11)
    for k in range(60):
        t, e = random_instance(rng, 6 + k % 30)
        if not any(e):
            continue
        km = K.kaplan_meier(t, e)
        for t0 in (1.0, 3.0, 6.0):
            s, var = greenwood_by_hand(t, e, t0)
            i = int(np.searchsorted(km["t"], t0, side="right")) - 1
            if i < 0:
                assert s == 1
                continue
            assert km["s"][i] == pytest.approx(float(s), abs=1e-12)
            assert km["s"][i] ** 2 * km["gw"][i] == pytest.approx(float(var), abs=1e-12)


def test_greenwood_standardfehler_gegen_bootstrap():
    """Bootstrap-Standardabweichung von Ŝ(4) (Ziehen der Paare mit Zurücklegen, numpy-Zufall) gegen den Greenwood-Standardfehler, fünf feste Stichproben, 15 % Toleranz."""
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        n = 250
        d = rng.lognormal(np.log(4.0), 0.9, n)
        c = rng.uniform(0, 12, n)
        obs, ev = np.minimum(d, c), d <= c
        km = K.kaplan_meier(obs, ev)
        i = int(np.searchsorted(km["t"], 4.0, side="right")) - 1
        greenwood_se = km["s"][i] * math.sqrt(km["gw"][i])
        boots = []
        for _ in range(1500):
            idx = rng.integers(0, n, n)
            kb = K.kaplan_meier(obs[idx], ev[idx])
            boots.append(K.survival_at(kb, 4.0))
        assert greenwood_se == pytest.approx(float(np.std(boots, ddof=1)), rel=0.15), seed


def test_loglog_band_gegen_handformel():
    """Band an t = 5 im Handbeispiel: log-log mit σ_LL = √(Σ d/(n(n−d))) / |ln S|."""
    km = K.kaplan_meier([2, 3, 4, 5, 5, 8], [1, 0, 1, 1, 0, 1])
    s = 5 / 12
    gw = 1 / 30 + 1 / 12 + 1 / 6
    sig = math.sqrt(gw) / abs(math.log(s))
    z = 1.959963984540054
    _, lo, hi = K.greenwood_band(km, 5.0, 0.95, "loglog")
    assert lo == pytest.approx(s ** math.exp(z * sig), abs=1e-6) and hi == pytest.approx(s ** math.exp(-z * sig), abs=1e-6)


# ------------------------------------------------------------------------------------------------ Orakel: mittlere Lebenszeit


def area_by_fractions(times, events, tau):
    """Fläche unter der Treppe bis tau in Brüchen, Treppe aus dem Umverteilungsverfahren."""
    surv = redistribute_survival(times, events)
    area, prev_t, prev_s = F(0), F(0), F(1)
    for x in sorted(surv):
        if x >= tau:
            break
        area += prev_s * (F(x) - prev_t)
        prev_t, prev_s = F(x), surv[x]
    return area + prev_s * (F(tau) - prev_t)


def test_mittlere_lebenszeit_gegen_flaeche_in_bruechen_und_gitter():
    rng = np.random.default_rng(5)
    for k in range(50):
        t, e = random_instance(rng, 8 + k % 25)
        if not any(e):
            continue
        tau = float(max(t)) * (0.4 + 0.6 * (k % 5) / 4)
        km = K.kaplan_meier(t, e)
        got = K.restricted_mean(km, tau)
        assert got == pytest.approx(float(area_by_fractions(t, e, tau)), abs=1e-10)
        grid = (np.arange(2000) + 0.5) * tau / 2000          # Mittelpunktsregel: Fehler höchstens Schrittweite mal Summe der Sprünge (<= 1)
        ys = np.array([K.survival_at(km, float(x)) for x in grid])
        assert got == pytest.approx(float(np.mean(ys)) * tau, abs=tau / 1000 + 1e-12)


# ------------------------------------------------------------------------------------------------ Orakel: die Wahrheit der Welt


def test_wahrheit_gegen_scipy():
    stats = pytest.importorskip("scipy.stats")
    integrate = pytest.importorskip("scipy.integrate")
    dist = stats.lognorm(s=0.9, scale=4.0)
    for t in (0.5, 2.0, 4.0, 8.0, 12.0, 30.0):
        assert float(S.true_survival(t)) == pytest.approx(float(dist.sf(t)), abs=1e-12)
    for q in (0.1, 0.25, 0.5, 0.9):
        assert S.true_quantile(q) == pytest.approx(float(dist.ppf(q)), rel=1e-9)
    assert S.true_mean() == pytest.approx(float(dist.mean()), rel=1e-12)
    for tau in (1.0, 6.0, 8.0, 30.0):
        want = integrate.quad(dist.sf, 0.0, tau)[0]
        assert S.true_restricted_mean(tau) == pytest.approx(want, rel=1e-8)


# ------------------------------------------------------------------------------------------------ Orakel: Unverzerrtheit mit anderem Zufallsstrom


def test_kaplan_meier_ist_unverzerrt_an_einer_welt_mit_numpy_zufall():
    """Mittel von Ŝ(t) über 200 Läufe (Fenster 8 Tage, n = 150, Welt mit numpy erzeugt) liegt im Band ±0,01 um die wahre Kurve; Zensierte als Ereignis liegt weit daneben."""
    stats = pytest.importorskip("scipy.stats")
    dist = stats.lognorm(s=0.9, scale=4.0)
    est = {t: [] for t in (2.0, 4.0, 6.0)}
    naive = {t: [] for t in est}
    for r in range(200):
        rng = np.random.default_rng(5000 + r)
        d = rng.lognormal(np.log(4.0), 0.9, 150)
        follow = 8.0 - rng.uniform(0, 8.0, 150)
        obs, ev = np.minimum(d, follow), d <= follow
        km = K.kaplan_meier(obs, ev)
        kn = K.kaplan_meier(obs, np.ones(150, bool))
        for t in est:
            est[t].append(K.survival_at(km, t))
            naive[t].append(K.survival_at(kn, t))
    for t in est:
        truth = float(dist.sf(t))
        assert abs(float(np.mean(est[t])) - truth) < 0.01, t
    assert abs(float(np.mean(naive[4.0])) - float(dist.sf(4.0))) > 0.1    # „Zensierte als Ereignis“: bei t = 4 liegt die Kurve weit unter der Wahrheit
