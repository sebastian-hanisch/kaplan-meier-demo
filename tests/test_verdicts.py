"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.
Ein Verdikt ist eine reine Funktion (Art, Text) von Zahlen; getestet werden die Zweige, nicht ein Seed."""

import math
from fractions import Fraction as F

import numpy as np

import kpm_algorithm as A
import kpm_constants as C
import kpm_core as K
import kpm_evaluation as E
import kpm_scenario as S


def _analysis(times, events, window=8, kind=C.KIND_WINDOW):
    sample = S.Sample(np.asarray(times, dtype=float), np.asarray(events, dtype=bool))
    return E.Analysis(E.Settings(sample.n, window, kind, 1), sample, A.all_ways(sample), sample.censored_share, C.MEDIAN_TRUE, S.true_mean())


def test_formate_komma_minus_und_nicht_schaetzbar():
    assert E.de(3.456) == "3,46" and E.de(float("nan")) == "nicht schätzbar" and E.de(2.0, 1) == "2,0"
    assert E.pct(-21.34) == "−21,3 %" and E.pct(0.04) == "+0,0 %" and E.pct(5.0, 0) == "+5 %"
    assert E.deviation_percent(3.0, 4.0) == -25.0 and math.isnan(E.deviation_percent(float("nan"), 4.0)) and math.isnan(E.deviation_percent(3.0, 0.0))
    assert E.fraction_text(F(5, 6)) == "5/6" and E.fraction_text(F(0)) == "0" and E.fraction_text(F(2)) == "2"


def test_verdict_hand_je_schritt():
    rows = E.hand_rows()
    assert E.verdict_hand(0, rows, 4)[1].startswith("Noch nichts verarbeitet")
    k1, t1 = E.verdict_hand(1, rows, 4)
    assert k1 == "info" and "Nach t = 2: 6 Container im Risiko, 1 Abholung, Faktor 5/6, S(2) = 5/6 = 0,8333." == t1
    t2 = E.verdict_hand(2, rows, 4)[1]
    assert "Nach t = 4: 4 Container im Risiko" in t2 and "S(4) = 5/8 = 0,6250." in t2 and "Median" not in t2     # 0,625 > 0,5
    t3 = E.verdict_hand(3, rows, 4)[1]
    assert "Faktor 2/3, S(5) = 5/12 = 0,4167." in t3 and t3.endswith("Dieses t ist der Median.")        # erster Schritt mit S <= 0,5
    t4 = E.verdict_hand(4, rows, 4)[1]
    assert "Nach t = 8: 1 Container im Risiko" in t4 and "S(8) = 0 = 0,0000." in t4 and t4.endswith("die Kurve ist am Ende.")


def test_verdict_core_zweige():
    # Handbeispiel-Daten als Stichprobe: Median Kaplan–Meier 5, Zensierte als Ereignis 4, nur Ereignisse 4; Mittel aller 4,5
    a = _analysis([2, 3, 4, 5, 5, 8], [1, 0, 1, 1, 0, 1])
    kind, text = E.verdict_core(a)
    assert "33,3 % der Container sind beim Ende der Beobachtung noch da." in text
    assert "Das Mittel der beobachteten Zeiten liegt bei 4,50 Tagen (wahres Mittel 6,00: −25,0 % gegen die Wahrheit)." in text      # (4,5 − 5,997)/5,997
    assert "Zensierte als Ereignis 4,00 (+0,0 %), nur Ereignisse 4,00 (+0,0 %), Kaplan–Meier 5,00 (+25,0 %); wahrer Median 4,00 Tage." in text
    assert kind == "info"          # Kaplan–Meier liegt hier weiter weg als die beiden anderen: kein „success“
    # nichts zensiert
    c = _analysis([2, 3, 4, 5], [1, 1, 1, 1])
    assert E.verdict_core(c)[0] == "info" and "Nichts ist zensiert" in E.verdict_core(c)[1] and "Median 3,00 Tage, Mittel 3,50 Tage" in E.verdict_core(c)[1]
    # keine Abholung
    d = _analysis([2, 3], [0, 0])
    kd, td = E.verdict_core(d)
    assert kd == "warning" and "keine einzige Abholung" in td and "(2,50 Tage)" in td
    # Median nicht schätzbar
    e = _analysis([1, 2, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0])
    ke, te = E.verdict_core(e)
    assert ke == "warning" and "erreicht 0,5 nicht (niedrigster Wert 0,833)" in te and "wahr sind 4,00" in te


def test_verdict_core_success_wenn_kaplan_meier_am_naechsten_liegt():
    a = _analysis([1.0, 2.0, 3.0, 3.9, 4.0, 4.1], [1, 1, 1, 1, 0, 0])
    kmw = a.km
    assert kmw.status == "ok"
    d_km = abs(E.deviation_percent(kmw.median, 4.0))
    d_all = abs(E.deviation_percent(a.ways[A.WAY_ALL].median, 4.0))
    d_ev = abs(E.deviation_percent(a.ways[A.WAY_EVENTS].median, 4.0))
    assert E.verdict_core(a)[0] == ("success" if d_km < min(d_all, d_ev) else "info")


def _rep(medians_all, medians_events, medians_km, means_all):
    return {"seeds": len(medians_all), "censored_share": 0.5, "ways": {
        A.WAY_ALL: {"median": medians_all, "mean": means_all},
        A.WAY_EVENTS: {"median": medians_events, "mean": [0.0] * len(medians_all)},
        A.WAY_KM: {"median": medians_km, "mean": [float("nan")] * len(medians_all)}}}


def test_verdict_ways_zahlen_und_nicht_definierte_laeufe():
    nan = float("nan")
    rep = _rep([2.0, 3.0, 4.0], [1.0, 2.0, 3.0], [3.5, 4.0, 4.5], [3.0, 3.0, 3.0])
    kind, text = E.verdict_ways(rep, 4.0, 6.0)
    assert kind == "info"
    assert "über 3 feste Zufalls-Seeds (wahrer Median 4,00)" in text
    assert "Zensierte als Ereignis 3,00 (Spanne 2,00 bis 4,00)" in text and "Nur Ereignisse 2,00 (Spanne 1,00 bis 3,00)" in text and "Kaplan–Meier 4,00 (Spanne 3,50 bis 4,50)" in text
    assert "im Median der Läufe bei 3,00 (wahres Mittel 6,00, −50,0 % gegen die Wahrheit)" in text
    rep2 = _rep([2.0, 3.0, 4.0], [1.0, 2.0, 3.0], [nan, 4.0, nan], [3.0, 3.0, 3.0])
    k2, t2 = E.verdict_ways(rep2, 4.0, 6.0)
    assert k2 == "warning" and "in 2 von 3 Läufen keinen Median" in t2 and "für die übrigen 1" in t2
    rep3 = _rep([2.0], [1.0], [nan], [3.0])
    assert "in keinem der 1 Läufe schätzbar" in E.verdict_ways(rep3, 4.0, 6.0)[1]


def test_verdict_band_zweige():
    rows = [{"t": 2, "inside": True}, {"t": 4, "inside": True}, {"t": 8, "inside": False}, {"t": 12, "inside": None}]
    kind, text = E.verdict_band("log-log", rows, (93.6, 95.9, 1000))
    assert kind == "info" and "an 2 von 3 Zeitpunkten (t = 2, 4, 8) im log-log-Band" in text
    assert "(1000 Läufe je Zelle, alle Zellen der Tabelle unten): 93,6 bis 95,9 % für das log-log-Band" in text
    assert E.verdict_band("einfach", rows[:2], None)[0] == "success" and "Gemessene" not in E.verdict_band("einfach", rows[:2], None)[1]
    kind0, text0 = E.verdict_band("log-log", [{"t": 2, "inside": None}], None)
    assert kind0 == "warning" and "an keinem der Zeitpunkte" in text0


def test_verdict_median_zweige():
    ok = _analysis([1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 0, 0])
    rows = E.quantile_rows(ok)
    k, t = E.verdict_median(ok, rows, 100.0)
    assert k == "success" and "Median 3,00 Tage (wahr 4,00)" in t and "Fenster 8 Tage, n = 300" in t and "in 100,0 % der Läufe schätzbar" in t
    assert "Fenster" not in E.verdict_median(ok, rows, None)[1]
    bad = _analysis([1, 2, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0])
    kb, tb = E.verdict_median(bad, E.quantile_rows(bad), 43.4)
    assert kb == "warning" and "niedrigster Wert 0,833" in tb and "10 %-Quantil: 1,00 Tage (wahr 1,26)" in tb and "in 43,4 % der Läufe" in tb
    none = _analysis([2, 3], [0, 0])
    assert E.verdict_median(none, E.quantile_rows(none), None)[0] == "warning"
    # kein tieferes Quantil schätzbar: Kurve fällt nie unter 0,9
    flat = _analysis([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1] + [0] * 11)
    kf, tf = E.verdict_median(flat, E.quantile_rows(flat), None)
    assert kf == "warning" and "Auch kein tieferes Quantil" in tf      # S(1) = 11/12 > 0,9: auch das 10-%-Quantil fehlt


def test_hilfsfunktionen_der_messung():
    assert E.spread([1.0, float("nan"), 3.0, 2.0]) == {"median": 2.0, "low": 1.0, "high": 3.0, "defined": 3, "of": 4}
    s = E.spread([float("nan")])
    assert s["defined"] == 0 and math.isnan(s["median"])
    assert E._join(["a"]) == "a" and E._join(["a", "b"]) == "a und b" and E._join(["a", "b", "c"]) == "a, b und c"
    km = K.kaplan_meier([1, 2, 3], [1, 0, 0])
    assert abs(E.lowest_survival(km) - 2 / 3) < 1e-12
    assert E.lowest_survival({"s": np.zeros(0)}) == 1.0
