"""Kaplan–Meier: zensierte Zeiten lesen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Eine Verweilzeit, die beim Ende der Beobachtung noch läuft, ist keine kurze Verweilzeit, sondern eine, von der man nur „mindestens so lang“ weiß.
Die Demo zeigt an erfundenen Containerdaten (die Wahrheit ist bekannt), wie weit das Mittel aller Zeiten und das Weglassen der noch anwesenden
Container danebenliegen und was Kaplan–Meier daraus macht. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import kpm_algorithm as A
import kpm_constants as C
import kpm_core as K
import kpm_evaluation as E
import kpm_scenario as S
from kpm_evaluation import Settings, de
from kpm_layout import dunkles_design, kopf_css
from kpm_pdf_export import build_pdf
from kpm_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kpm_visualization import build_band, build_defined, build_hand, build_rmst, build_ways

st.set_page_config(page_title="Kaplan–Meier – Sebastian Hanisch", layout="wide")


def dl(x: float) -> str:
    """Differenz für `st.metric(delta=...)`: mit ASCII-Minus, denn nur dann zeigt Streamlit den Pfeil nach unten."""
    return E.signed(x).replace("−", "-")


@st.cache_data(show_spinner=False)
def _analysis(settings: Settings):
    return E.analyse(settings)


@st.cache_data(show_spinner=False)
def _repeat(kind: str, window: int, n: int):
    return E.repeat_summary(kind, window, n)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kpm_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⏳ Kaplan–Meier: zensierte Zeiten lesen")
st.markdown(
    """
Ein Container im Block wird irgendwann abgeholt; die **Verweilzeit** ist die Zeit bis dahin. Endet die Beobachtung, **ist ein Teil der Container noch da**: von ihnen weiß man nur
„mindestens so lang“. Das **Mittel aller beobachteten Zeiten** und das **Weglassen der noch anwesenden Container** behandeln sie falsch. **Kaplan–Meier** benutzt jeden Container genau bis zu dem Tag,
an dem seine Beobachtung endet, und liest daraus eine Kurve: den Anteil, der nach t Tagen noch da ist.

Die Daten sind **erfunden** (Simulation, die Wahrheit ist bekannt). Der Fall ohne Zensierung steht in der [verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo).
Weiter unten: vier **🔬 Experimente** (von Hand, viele Stichproben, Greenwood-Band, Median nicht schätzbar), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Kaplan–Meier", expanded=True):
    st.markdown(
        """
1. **Ereigniszeiten ordnen.** Nur Tage, an denen mindestens ein Container abgeholt wurde, zählen. Dort gilt: **n** Container waren bis dahin noch da (im Risiko), **d** davon wurden abgeholt.
2. **Faktor und Produkt.** Der Anteil, der den Tag übersteht, ist (n − d)/n. Die Kurve S(t) ist das Produkt aller Faktoren bis t: Sie fällt nur an Ereigniszeiten.
3. **Zensierte bleiben im Risiko, bis ihre Beobachtung endet.** Danach fehlen sie in n. So geht ihre Information („mindestens so lang“) ein, ohne dass man eine Abholzeit für sie erfinden muss.
4. **Median:** das kleinste t mit S(t) ≤ 0,5. **Gleichstand:** zeigt ein Tag eine Abholung und eine Zensierung, zählt das Ereignis zuerst; der Zensierte bleibt an diesem Tag im Risiko.
5. **Warum das funktioniert:** Jeder Faktor ist ein Anteil unter denen, die den Tag noch erreicht haben. Wer zensiert wurde, hat bis dahin überlebt und zählt bis dahin mit, danach gar nicht.
        """
    )

st.info(
    "**Annahme in dieser Demo: Die Zensierung ist unabhängig vom Ereignis.** Ob ein Container beim Ende noch da ist, hängt hier nur davon ab, wann er ankam (Fenster) bzw. von der festen Frist, "
    "nicht davon, wie bald er abgeholt worden wäre. Das gilt, weil die Daten so gebaut sind; in echten Daten ist es eine Annahme, die Kaplan–Meier nicht prüft."
)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Beobachtung**")
    kind = st.selectbox(
        "Zensierungsart", C.KINDS, key="kind_select",
        help="Beobachtungsfenster: Die Container kommen nacheinander in den ersten W Tagen an, die Beobachtung endet für alle am Tag W; wer spät ankam, wurde nur kurz beobachtet. "
             "Feste Frist (Typ I): jeder Container wird genau W Tage nach seiner Ankunft beobachtet.",
    )
    window = st.slider("Länge des Fensters bzw. der Frist (Tage)", *bounds("window_slider"), key="window_slider",
                       help="Fenster: Ankünfte gleichverteilt in den ersten W Tagen, Ende am Tag W. Feste Frist: jeder Container wird W Tage beobachtet.")
    st.markdown("**Stichprobe**")
    n = st.slider("Zahl der Container n", *bounds("n_slider"), key="n_slider", help="Wie viele Container beobachtet werden (die Wahrheit bleibt gleich).")
    seed = st.number_input("Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Stichprobe generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"kind_select": kind, "window_slider": int(window), "n_slider": int(n), "seed_input": int(seed)})

settings = Settings(int(n), int(window), kind, int(seed))
try:
    a = _analysis(settings)
except (K.SurvivalError, ValueError) as err:
    st.error(f"Die Stichprobe konnte nicht ausgewertet werden: {err}")
    st.stop()

# --- Von Hand ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Von Hand, sechs Container")
hand_rows = E.hand_rows()
hand = E.hand_summary()
st.markdown(
    "Beobachtete Zeiten in Tagen: **2, 3+, 4, 5, 5+, 8** (ein **+** heißt: beim Ende der Beobachtung noch da). Die Kurve wächst mit jeder verarbeiteten Ereigniszeit; "
    "der Regler geht die vier Ereigniszeiten (t = 2, 4, 5, 8) der Reihe nach durch."
)
step = st.slider("Verarbeitete Ereigniszeiten", 0, len(hand_rows), 3, key="hand_step")
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(build_hand(hand_rows, step), width="stretch", key="hand_chart")
with c2:
    kind_h, text_h = E.verdict_hand(step, hand_rows, len(hand_rows))
    getattr(st, kind_h)(text_h)
    if step > 0:
        st.dataframe(
            [{"t": r["t"], "n": r["n"], "d": r["d"], "(n − d)/n": E.fraction_text(r["factor"]), "S(t)": E.fraction_text(r["s"]), "dezimal": de(float(r["s"]), 4)} for r in hand_rows[:step]],
            hide_index=True, width="stretch",
        )
        st.caption("n: im Risiko, d: abgeholt, S(t): Produkt der Faktoren bis t.")
st.caption(
    "Bei t = 5 stehen eine Abholung und eine Zensierung (5+): Das Ereignis zählt zuerst, der Zensierte bleibt an diesem Tag im Risiko (n = 3). "
    f"Median: {hand['median']} Tage. Mittel aller Zeiten {de(float(hand['mean_all']))}, Mittel nur der Ereignisse {de(float(hand['mean_events']), 2)} ({E.fraction_text(hand['mean_events'])}). "
    f"Mittlere Lebenszeit bis {hand['tau']} Tage: {E.fraction_text(hand['rmst'])} = {de(float(hand['rmst']))}. "
    f"Greenwood-Varianz bei t = {hand['median']}: {E.fraction_text(hand['greenwood_var'])} = {de(float(hand['greenwood_var']), 4)}, Standardfehler {de(float(hand['greenwood_var']) ** 0.5, 3)}."
)

st.markdown("---")

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: drei Wege auf derselben Stichprobe")
st.markdown(
    "- **Zensierte als Ereignis** (das ist dasselbe wie das **Mittel aller beobachteten Zeiten**): Jeder noch anwesende Container zählt, als wäre er an dem Tag abgeholt worden, an dem seine Beobachtung endet.\n"
    "- **Nur Ereignisse**: Wer beim Ende noch da ist, wird weggelassen.\n"
    "- **Kaplan–Meier**: Die Zensierten bleiben bis zu ihrem Beobachtungsende im Risiko."
)
allw, evw, kmw = a.ways[A.WAY_ALL], a.ways[A.WAY_EVENTS], a.km
m1, m2, m3, m4 = st.columns(4)
m1.metric("Zensiert", f"{de(100 * a.censored_share, 1)} %", help=f"Anteil der Container, die beim Ende der Beobachtung noch da sind; {a.sample.n_events} von {a.sample.n} Abholungen gesehen.")
m2.metric("Mittel aller Zeiten", f"{de(allw.mean)} Tage", delta=f"{dl(allw.mean - a.true_mean)} gegen Wahrheit", delta_color="off", help=f"Wahres Mittel: {de(a.true_mean)} Tage.")
m3.metric("Median: als Ereignis", f"{de(allw.median)} Tage", delta=f"{dl(allw.median - a.true_median)} gegen Wahrheit", delta_color="off",
          help=f"Median der Kurve „Zensierte als Ereignis“. Wahrer Median: {de(a.true_median)} Tage.")
if kmw.status == "ok":
    m4.metric("Median: Kaplan–Meier", f"{de(kmw.median)} Tage", delta=f"{dl(kmw.median - a.true_median)} gegen Wahrheit", delta_color="off", help=f"Wahrer Median: {de(a.true_median)} Tage.")
else:
    m4.metric("Median: Kaplan–Meier", "kein Median", delta="Kurve erreicht 0,5 nicht" if kmw.status == "nicht_schaetzbar" else "keine Abholung gesehen", delta_color="off")
st.caption(f"Wahrheit in dieser Welt: Mittel {de(a.true_mean)} Tage, Median {de(a.true_median)} Tage. Die Zahl unter dem Wert ist die Abweichung von der Wahrheit.")
st.plotly_chart(build_ways(a), width="stretch", key="ways_chart")
st.caption("Gepunktet: die wahre Kurve. Senkrechte Striche auf der Kaplan–Meier-Kurve: Container, die beim Ende noch da sind (höchstens 120 eingezeichnet). Gestrichelt: S = 0,5, dort liegt der Median.")

if st.button("📄 PDF erstellen", key="make_pdf", help="Fasst die aktuelle Stichprobe und die Läufe über 30 feste Zufalls-Seeds zusammen."):
    st.session_state["pdf_bytes"] = build_pdf(a, _repeat(settings.kind, settings.window, settings.n))
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="kaplan-meier.pdf", mime="application/pdf", key="download_pdf")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie weit liegen die Wege daneben?")
kind_v, text_v = E.verdict_core(a)
getattr(st, kind_v)(text_v)

st.markdown("---")

# --- Experiment 1: Wege gegen die Wahrheit ------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Die Wege gegen die Wahrheit, über viele Stichproben")
rep = _repeat(settings.kind, settings.window, settings.n)
true_mean = a.true_mean
st.caption(f"Dieselben Einstellungen wie in der Seitenleiste, {rep['seeds']} feste Zufalls-Seeds (nie eine einzelne Stichprobe als Befund). Median, kleinster und größter Wert über die Läufe.")
rows_rep = []
for name in A.WAYS:
    sm = E.spread(rep["ways"][name]["median"])
    rows_rep.append({"Weg": name, "Median der Läufe": de(sm["median"]), "Spanne": f"{de(sm['low'])} bis {de(sm['high'])}" if sm["defined"] else "-",
                     "Läufe mit Median": f"{sm['defined']} von {sm['of']}", "gegen Wahrheit": E.pct(E.deviation_percent(sm["median"], a.true_median))})
st.dataframe(rows_rep, hide_index=True, width="stretch")
mean_sp = E.spread(rep["ways"][A.WAY_ALL]["mean"])
st.caption(f"Mittel aller beobachteten Zeiten: {de(mean_sp['median'])} im Median der Läufe ({de(mean_sp['low'])} bis {de(mean_sp['high'])}), wahres Mittel {de(true_mean)}. "
           "Kaplan–Meier hat in der Regel kein Mittel der Verweilzeit: Ist die größte beobachtete Zeit zensiert, endet die Kurve oberhalb von null; die App meldet das Mittel immer als nicht definiert (Experiment 4: mittlere Lebenszeit bis τ).")
kind_w, text_w = E.verdict_ways(rep, a.true_median, true_mean)
getattr(st, kind_w)(text_w)

meas = E.measurements()
if meas:
    st.markdown("**Eingefrorene Messung: Fenster und feste Frist bei gleicher Länge** (n = 300, 1000 Läufe je Zeile, Mittel mit Standardfehler über die Läufe; "
                f"wahrer Median {de(meas['meta']['median_true'])}, wahres Mittel {de(meas['meta']['mean_true'])})")

    def _cell(d: dict) -> str:
        return "nicht definiert" if d["mean"] is None else f"{de(d['mean'])} ± {de(d['se'])}"

    short = {C.KIND_WINDOW: "Fenster", C.KIND_FIXED: "Feste Frist"}
    st.dataframe(
        [{"Art": short[c["kind"]], "Länge (Tage)": str(c["length"]), "zensiert": f"{de(100 * c['censored']['mean'], 1)} %", "Mittel aller": _cell(c["mean_all"]), "Mittel nur Ereignisse": _cell(c["mean_events"])}
         for c in meas["cells"]],
        hide_index=True, width="stretch",
    )
    st.dataframe(
        [{"Art": short[c["kind"]], "Länge (Tage)": str(c["length"]), "Median: als Ereignis": _cell(c["median_all"]), "Median: nur Ereignisse": _cell(c["median_events"]),
          "Median: Kaplan–Meier": _cell(c["median_km"]), "ohne KM-Median": f"{de(100 * c['undef_km'], 1)} %"} for c in meas["cells"]],
        hide_index=True, width="stretch",
    )
    st.caption("Mittel über die Läufe, in denen der Wert definiert ist (die Spalte „ohne KM-Median“ zählt beim Kaplan–Meier-Median die übrigen). " + E.cells_sentence(meas))
    st.markdown("**Mehr Daten reparieren das Mittel nicht** (Fenster 30 Tage, 1000 Läufe je Zeile):")
    st.dataframe(
        [{"n": str(c["n"]), "zensiert": f"{de(100 * c['censored']['mean'], 1)} %", "Mittel aller (wahr " + de(meas["meta"]["mean_true"]) + ")": _cell(c["mean_all"]), "Median Kaplan–Meier (wahr 4,00)": _cell(c["median_km"])}
         for c in meas["v1"]],
        hide_index=True, width="stretch",
    )
else:
    st.warning("Die eingefrorene Messung (tools/messung_kaplan_meier.json) fehlt: Die Tabellen dieses Abschnitts entfallen.")

st.markdown("---")

# --- Experiment 2: Greenwood-Band ---------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wie sicher ist die Kurve? Das Greenwood-Band")
st.markdown("Die Kurve ist geschätzt, nicht bekannt. Die **Greenwood-Formel** schätzt ihre Streuung aus den Zahlen n und d an den Ereigniszeiten; daraus entsteht ein punktweises Band zum nominalen Niveau 95 %. "
            "Zwei Fassungen: **log-log** (bleibt zwischen 0 und 1) und **einfach** (Ŝ ± z·Standardfehler).")
method_label = st.radio("Art des Bandes", ["log-log", "einfach"], horizontal=True, key="band_method")
method = "loglog" if method_label == "log-log" else "plain"
st.plotly_chart(build_band(a, method), width="stretch", key="band_chart")
st.caption("Punkte: die wahre Kurve an t = 2, 4, 8, 12 (grün: im Band, rote Kreuze: außerhalb). Das Band gilt punktweise, nicht als Streifen über die ganze Kurve.")
checks = E.band_checks(a, method)
if checks:
    st.dataframe(
        [{"t": str(r["t"]), "wahres S(t)": de(r["truth"], 3), "geschätztes S(t)": de(r["s"], 3), "Band": f"{de(r['low'], 3)} bis {de(r['high'], 3)}" if r["inside"] is not None else "kein Band",
          "wahre Kurve im Band": "-" if r["inside"] is None else ("ja" if r["inside"] else "nein")} for r in checks],
        hide_index=True, width="stretch",
    )
kind_b, text_b = E.verdict_band(method_label, checks, E.coverage_range(method))
getattr(st, kind_b)(text_b)
if meas.get("coverage"):
    st.markdown(f"**Eingefrorene Messung: Überdeckung des nominalen 95-%-Bandes** ({meas['meta']['cover_runs']} Läufe je Zelle, Beobachtungsfenster, wahre Kurve lognormal; „im Band“ heißt: die wahre S(t) liegt zwischen den Grenzen)")
    st.dataframe(
        [{"Fenster (Tage)": str(c["length"]), "n": str(c["n"]), "t": str(r["t"]), "wahres S(t)": de(r["true_s"], 3), "log-log (%)": de(r["loglog"], 1), "einfach (%)": de(r["plain"], 1),
          "Läufe ohne Band": str(r["loglog_undef"])} for c in meas["coverage"] for r in c["rows"]],
        hide_index=True, width="stretch",
    )

st.markdown("---")

# --- Experiment 3: Median nicht schätzbar ---------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Wenn der Median nicht schätzbar ist")
st.markdown("Endet die Beobachtung zu früh, **erreicht die Kurve 0,5 vielleicht nicht**: Dann gibt es keinen Median, und das Mittel der Mediane aus den Läufen, in denen es klappt, gilt nur für diese Läufe. "
            "Verändern Sie die Länge des Fensters in der Seitenleiste oder laden Sie den Schnellstart „Fenster 4 Tage“.")
qrows = E.quantile_rows(a)
defined_pct = None
window_rows = meas.get("defined", []) if meas else []
if settings.kind == C.KIND_WINDOW and settings.n == 300:
    match = [r for r in window_rows if r["length"] == settings.window]
    defined_pct = 100.0 * (1.0 - match[0]["undef_km"]) if match else None
kind_m, text_m = E.verdict_median(a, qrows, defined_pct)
getattr(st, kind_m)(text_m)
st.dataframe(
    [{"Quantil": f"{int(round(100 * r['q']))} %", "wahr (Tage)": de(r["truth"]), "Kaplan–Meier (Tage)": de(r["estimate"])} for r in qrows],
    hide_index=True, width="stretch",
)
st.caption("Das 10-%-Quantil ist die Zeit, bis 10 % der Container abgeholt sind. Die Kurve erreicht 0,9 früher als 0,5: Ist der Median schätzbar, ist es auch das 10-%-Quantil, aber nicht umgekehrt.")
tau = st.slider("τ für die mittlere Lebenszeit (Tage)", C.TAU_MIN, C.TAU_MAX, C.DEFAULT_TAU, key="tau_slider",
                help="Fläche unter der Kaplan–Meier-Kurve bis τ. τ darf die größte beobachtete Zeit nicht überschreiten.")
rm, rm_status = A.restricted_mean_checked(a.sample, float(tau))
true_rm = S.true_restricted_mean(float(tau))
r1, r2, r3 = st.columns(3)
r1.metric(f"Wahre mittlere Lebenszeit bis τ = {tau}", f"{de(true_rm)} Tage")
if rm_status == "ok":
    naive_rm = float(a.sample.obs.clip(max=float(tau)).mean())
    r2.metric("Kaplan–Meier", f"{de(rm)} Tage", delta=f"{dl(rm - true_rm)} gegen Wahrheit", delta_color="off")
    r3.metric("Zensierte als Ereignis (bei τ gekappt)", f"{de(naive_rm)} Tage", delta=f"{dl(naive_rm - true_rm)} gegen Wahrheit", delta_color="off")
    st.plotly_chart(build_rmst(a, float(tau)), width="stretch", key="rmst_chart")
elif rm_status == "tau_ueber_beobachtung":
    r2.metric("Kaplan–Meier", "nicht schätzbar", delta="τ liegt hinter der größten Beobachtungszeit", delta_color="off")
    st.warning(f"τ = {tau} liegt hinter der größten beobachteten Zeit ({de(float(a.sample.obs.max()))} Tage): Dort gibt es keine Beobachtung, und die Fläche wäre nur fortgeschrieben. Wählen Sie ein kleineres τ.")
else:
    r2.metric("Kaplan–Meier", "nicht schätzbar", delta="keine Abholung gesehen", delta_color="off")
if window_rows:
    st.markdown("**Eingefrorene Messung: Wie oft ist der Kaplan–Meier-Median schätzbar?** (Beobachtungsfenster, n = 300, "
                f"{window_rows[0]['runs']} Läufe je Fenster; der Wert in der Spalte „Median im Mittel“ gilt nur für die Läufe, in denen es einen Median gibt)")
    st.plotly_chart(build_defined(window_rows, settings.window if settings.kind == C.KIND_WINDOW else None), width="stretch", key="defined_chart")
    st.dataframe(
        [{"Fenster (Tage)": str(r["length"]), "zensiert": f"{de(100 * r['censored']['mean'], 1)} %", "Läufe mit Median": f"{de(100 * (1 - r['undef_km']), 1)} %",
          "Median im Mittel (wahr 4,00)": f"{de(r['median_km']['mean'])} ± {de(r['median_km']['se'])}"} for r in window_rows],
        hide_index=True, width="stretch",
    )
    n_rows = meas.get("defined_n", [])
    if n_rows:
        parts = ", ".join(f"n = {r['n']}: {de(100 * (1 - r['undef_km']), 1)} %" for r in n_rows)
        st.caption(f"Bei Fenster 4 Tage ist der Median schätzbar in (je {n_rows[0]['runs']} Läufe) {parts} der Läufe (Binomial-Standardfehler je Wert rund 1,6 Prozentpunkte). "
                   "Das Fenster von 4 Tagen ist so lang wie der wahre Median; die Kurve endet dann bei rund 0,5 (Deutung, nicht gemessen). "
                   "Der Anteil bleibt mit n nur bei diesem Fenster fast gleich; bei längerem Fenster wächst er mit n, bei kürzerem fällt er (Prüfrechnung, siehe README).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
cover_txt = E.coverage_sentence()
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Zensierung ist unabhängig vom Ereignis** | Scheiden Container kurz vor ihrer Abholung aus der Beobachtung aus, würden sie als „noch da“ gelesen, und die Kurve fiele zu langsam (die Richtung folgt aus der Rechnung, die Größe ist in dieser Demo nicht gemessen). | Nicht in dieser Demo |
| **Die Beobachtung reicht über den Median hinaus** | Sonst kann die Kurve 0,5 verfehlen: Der Median ist dann nicht schätzbar (Experiment 4). | Quantil oder mittlere Lebenszeit bis τ |
| **Ein 95-%-Band hält 95 %** | Gemessene Überdeckung bei nominal 95 %: {cover_txt} | Log-log-Band |
| **Eine Welt: lognormal, Median 4, σ = 0,9** | Alle Zahlen gelten nur für diese Welt und für die zwei Zensierungsarten; andere Verteilungen sind nicht gemessen. | - |
| **Alle Container gleich (keine Merkmale)** | Unterschiede nach Kunde oder Containerart gehen in eine gemeinsame Kurve ein. | Nicht in dieser Demo |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen**
- Die Verweilzeit ist lognormal (Median 4 Tage, σ = 0,9); die Daten sind erfunden, kein Echtdatenbezug.
- Die Zensierung ist unabhängig vom Ereignis (siehe oben); die Verweilzeiten sind stetig verteilt, Gleichstände zwischen Abholungen treten praktisch nicht auf (bei der festen Frist stehen alle Zensierten genau bei der Frist).
- Beim Fenster kommen die Container gleichverteilt in den ersten W Tagen an.

**Grenzen**
- Eine Verteilungsfamilie, zwei Zensierungsarten, jeweils unabhängig. Kein Typ II (Stopp nach r Abholungen), keine Zensierung von links, keine Intervalle.
- Die Bänder sind punktweise (nicht als Streifen über die ganze Kurve) und nur an den angegebenen Zellen gemessen.
- Der Kaplan–Meier-Code ist selbst geschrieben und gegen eine exakte Bruchrechnung und ein Umverteilungsverfahren geprüft, nicht gegen ein Fremdpaket (`lifelines` ist nicht Teil des Test-Setups).

**Bekannte Probleme und Unsicherheiten**
- Die Läufe über 30 feste Zufalls-Seeds zeigen eine Spanne, keine Konfidenzgrenzen; die eingefrorene Messung nennt den Standardfehler des Mittels.
- Die Plotly-Diagramme sind im Browser angesehen, aber nicht mit allen Geräten und Designs getestet.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist für jeden von $n$ Containern eine Beobachtungszeit $T_i$ und ein Kennzeichen $\delta_i$ (1: Abholung gesehen, 0: beim Ende noch da). Gesucht ist die Überlebensfunktion $S(t)$,
der Anteil, der länger als $t$ Tage bleibt, und daraus der Median.

| Symbol | Bedeutung |
|---|---|
| $T_i$, $\delta_i$ | Beobachtungszeit und Kennzeichen (1 = Ereignis, 0 = zensiert) von Container $i$ |
| $t_1 < \dots < t_m$ | die verschiedenen Zeiten mit mindestens einem Ereignis |
| $n_j$ | Zahl der Container im Risiko bei $t_j$: $n_j = \#\{i : T_i \ge t_j\}$ |
| $d_j$ | Zahl der Ereignisse bei $t_j$ |
| $\hat S(t)$ | Kaplan–Meier-Schätzung von $S(t)$ |
| $z$ | Normalquantil, $z = 1{,}96$ für das nominale Niveau 95 % |
| $\tau$ | Obergrenze der mittleren Lebenszeit |

**Schätzer (Produkt-Grenzwert).**

$$\hat S(t) = \prod_{t_j \le t} \left(1 - \frac{d_j}{n_j}\right)$$

**Median** (kleinste Ereigniszeit, bei der die Kurve höchstens 0,5 erreicht; gibt es keine, ist er nicht schätzbar):

$$\hat t_{0{,}5} = \min\{\, t_j : \hat S(t_j) \le 0{,}5 \,\}$$

**Greenwood-Varianz.** Terme mit $n_j = d_j$ entfallen (dort ist $\hat S = 0$):

$$\widehat{\mathrm{Var}}\,\hat S(t) = \hat S(t)^2 \sum_{t_j \le t} \frac{d_j}{n_j\,(n_j - d_j)}$$

**Band, einfach** (auf $[0,1]$ gekappt):

$$\hat S(t) \pm z \, \hat S(t) \sqrt{\textstyle\sum_{t_j \le t} \frac{d_j}{n_j (n_j - d_j)}}$$

**Band, log-log** (bleibt in $[0,1]$), mit $\hat\sigma_{LL}(t) = \sqrt{\textstyle\sum_{t_j \le t} \frac{d_j}{n_j (n_j - d_j)}} \,/\, |\ln \hat S(t)|$:

$$\hat S(t)^{\exp(\pm z\,\hat\sigma_{LL}(t))}$$

**Mittlere Lebenszeit bis $\tau$** (Fläche unter der Treppe; $\tau$ höchstens die größte Beobachtungszeit):

$$\hat\mu(\tau) = \int_0^{\tau} \hat S(t)\, dt$$

**Die Welt dieser Demo.** Verweilzeit $D$ lognormal mit Median 4 und $\sigma = 0{,}9$:

$$S(t) = 1 - \Phi\!\left(\frac{\ln t - \ln 4}{0{,}9}\right), \qquad E[D] = 4\, e^{0{,}9^2/2} \approx 5{,}997$$

Beobachtet wird $T_i = \min(D_i, C_i)$ mit $\delta_i = 1$ genau dann, wenn $D_i \le C_i$. Beim Fenster ist $C_i$ = Fensterlänge minus Ankunftstag (gleichverteilt), bei der festen Frist ist $C_i$ = Frist.
Die wahre mittlere Lebenszeit ist $\int_0^\tau S(t)\,dt = \tau\,S(\tau) + E[D\,\mathbf 1(D \le \tau)]$.

**Literatur (nicht eingesehen, aus dem Gedächtnis, vor jeder Zitierung im Original zu prüfen).** E. L. Kaplan, P. Meier (1958), Nonparametric estimation from incomplete observations, Journal of the American Statistical Association 53 (Seitenzahlen nicht geprüft);
Klein, Moeschberger, Survival Analysis (Springer, 2. Auflage 2003), Kapitel zu Zensierung und Kaplan–Meier (Suchtreffer, nicht gelesen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
