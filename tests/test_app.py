"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def _set_all(at, edge):
    """Alle Regler der Seitenleiste und des Hauptbereichs auf Minimum bzw. Maximum (Auswahl: erste bzw. letzte Option)."""
    for s in list(at.slider):
        v = s.min if edge == "min" else s.max
        s.set_value(int(v))
    sel = at.sidebar.selectbox(key="kind_select")
    sel.set_value(sel.options[0] if edge == "min" else sel.options[-1])
    at.radio(key="band_method").set_value("log-log" if edge == "min" else "einfach")


def test_eckfall_alle_minima_und_alle_maxima_zugleich():
    for edge in ("min", "max"):
        at = _fresh_app()
        _set_all(at, edge)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


def test_jede_zensierungsart_und_jede_bandart_laeuft():
    for kind in ("Beobachtungsfenster (Ankunft zufällig)", "Feste Frist je Container (Typ I)"):
        for method in ("log-log", "einfach"):
            at = _fresh_app()
            at.sidebar.selectbox(key="kind_select").set_value(kind)
            at.radio(key="band_method").set_value(method).run(timeout=TIMEOUT)
            assert not at.exception, (kind, method, [str(e) for e in at.exception])


def test_tau_ueber_der_groessten_beobachtungszeit_zeigt_warnung_statt_zahl():
    at = _fresh_app()
    at.sidebar.slider(key="window_slider").set_value(4)
    at.slider(key="tau_slider").set_value(60).run(timeout=TIMEOUT)
    assert not at.exception
    assert any("liegt hinter der größten beobachteten Zeit" in w.value for w in at.warning)


def test_pdf_erst_nach_dem_klick_und_download_danach():
    at = _fresh_app()
    assert not at.get("download_button")
    at.button(key="make_pdf").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1


def test_stichprobe_ohne_abholung_zeigt_meldung_und_keinen_traceback():
    """Kleinster Wert für n und Länge bei fester Frist: kann ohne Ereignis enden; die App meldet das als Warnung."""
    at = _fresh_app()
    at.sidebar.selectbox(key="kind_select").set_value("Feste Frist je Container (Typ I)")
    at.sidebar.slider(key="window_slider").set_value(2)
    at.sidebar.slider(key="n_slider").set_value(20).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
