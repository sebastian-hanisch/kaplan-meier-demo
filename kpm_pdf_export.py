"""PDF-Zusammenfassung der aktuellen Einstellung (fpdf2, Kernschrift Helvetica: Umlaute gehen, Gedankenstrich, Minuszeichen und € nicht).

Die App erzeugt das PDF erst nach dem Klick auf „PDF erstellen“; hier steht nur der Aufbau aus fertigen Zahlen."""

from fpdf import FPDF

import kpm_algorithm as A
import kpm_evaluation as E


def _plain(text: str) -> str:
    """Kernschrift-sicher: Minuszeichen und Gedankenstrich werden zu '-', Markdown-Sterne fallen weg."""
    return text.replace("−", "-").replace("–", "-").replace("**", "")


def build_pdf(a: E.Analysis, rep: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    def line(text: str, size: int = 10, bold: bool = False, gap: float = 5.5):
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.multi_cell(width, gap, _plain(text), new_x="LMARGIN", new_y="NEXT")

    st = a.settings
    line("Kaplan-Meier: zensierte Zeiten lesen", 15, True, 8)
    line("Sebastian Hanisch - Demo, erfundene Daten (Verweilzeit lognormal, Median 4 Tage, Sigma 0,9)", 9)
    pdf.ln(2)
    line("Einstellung", 11, True)
    line(f"Zensierungsart: {st.kind}; Länge: {st.window} Tage; n = {st.n} Container; Zufalls-Seed {st.seed}.")
    line(f"Zensiert (beim Ende noch da): {E.de(100 * a.censored_share, 1)} % von {a.sample.n} Containern ({a.sample.n_events} Abholungen gesehen).")
    pdf.ln(2)
    line("Schätzungen dieser Stichprobe", 11, True)
    for name in A.WAYS:
        w = a.ways[name]
        mean = E.de(w.mean) if name != A.WAY_KM else "nicht definiert (die Kurve endet vor 0)"
        line(f"{name}: Median {E.de(w.median)} Tage, Mittel {mean}.")
    line(f"Wahrheit: Median {E.de(a.true_median)} Tage, Mittel {E.de(a.true_mean)} Tage.")
    pdf.ln(2)
    line("Kernfrage", 11, True)
    line(E.verdict_core(a)[1])
    pdf.ln(2)
    line(f"Über {rep['seeds']} feste Zufalls-Seeds", 11, True)
    line(E.verdict_ways(rep, a.true_median, a.true_mean)[1])
    pdf.ln(2)
    line("Annahme: Die Zensierung ist unabhängig vom Ereignis (hier durch die Bauweise der Daten erfüllt). Alle Zahlen stammen aus der Simulation dieser Demo.", 9)
    return bytes(pdf.output())
