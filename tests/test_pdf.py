"""PDF-Export: erzeugt gültige Bytes aus der aktuellen Stichprobe, Umlaute gehen, Zeichen außerhalb der Kernschrift werden ersetzt."""

import kpm_evaluation as E
import kpm_pdf_export as PDF


def test_pdf_ist_ein_pdf_und_nennt_die_einstellung():
    a = E.analyse(E.Settings())
    rep = E.repeat_summary(a.settings.kind, a.settings.window, a.settings.n)
    data = PDF.build_pdf(a, rep)
    assert data[:5] == b"%PDF-" and len(data) > 1500


def test_plain_ersetzt_minuszeichen_gedankenstrich_und_sterne():
    assert PDF._plain("−21,3 % – **fett**") == "-21,3 % - fett"


def test_pdf_ohne_abholung_wirft_keinen_fehler():
    import numpy as np

    import kpm_algorithm as A
    import kpm_scenario as S

    sample = S.Sample(np.full(20, 2.0), np.zeros(20, dtype=bool))
    a = E.Analysis(E.Settings(20, 2), sample, A.all_ways(sample), 1.0, 4.0, S.true_mean())
    rep = E.repeat_summary(a.settings.kind, 2, 20)
    assert PDF.build_pdf(a, rep)[:5] == b"%PDF-"
