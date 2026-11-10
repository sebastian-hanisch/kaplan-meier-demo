"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; die Aussagen der Preset-Hilfen gelten für den Zufalls-Seed des Presets."""

import kpm_algorithm as A
import kpm_constants as C
import kpm_evaluation as ev
import kpm_presets as P


def _analysis(name):
    p = C.PRESETS[name]
    return ev.analyse(ev.Settings(p["n"], p["window"], p["kind"], p["seed"]))


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.W_MIN <= preset["window"] <= C.W_MAX and 0 <= preset["seed"] <= C.SEED_MAX
        assert preset["kind"] in C.KINDS


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Fenster 30 Tage"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_the_default_is_a_choice():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert P.SETTING_SPECS["kind_select"].default in C.KINDS and P.SETTING_SPECS["kind_select"].choices == C.KINDS


def test_preset_namen_stimmen_mit_den_werten_ueberein():
    assert C.PRESETS["Fenster 30 Tage"]["window"] == 30 and C.PRESETS["Fenster 30 Tage"]["kind"] == C.KIND_WINDOW
    assert C.PRESETS["Fenster 8 Tage"]["window"] == 8 and C.PRESETS["Fenster 8 Tage"]["kind"] == C.KIND_WINDOW
    assert C.PRESETS["Fenster 4 Tage"]["window"] == 4 and C.PRESETS["Fenster 4 Tage"]["kind"] == C.KIND_WINDOW
    assert C.PRESETS["Feste Frist 8 Tage"]["window"] == 8 and C.PRESETS["Feste Frist 8 Tage"]["kind"] == C.KIND_FIXED


def test_preset_fenster_4_tage_hat_bei_seinem_seed_keinen_median():
    """Die Hilfe sagt „Bei diesem Zufalls-Seed ist der Median nicht schätzbar“: das gilt für Kaplan–Meier, nicht für die beiden anderen Wege."""
    a = _analysis("Fenster 4 Tage")
    assert a.km.status == "nicht_schaetzbar"
    assert a.ways[A.WAY_ALL].status == "ok"
    assert a.censored_share > 0.75          # „mehr als drei Viertel der Container sind zensiert“


def test_die_anderen_presets_haben_einen_median_und_die_zensierungsanteile_der_hilfe():
    for name in ("Fenster 30 Tage", "Fenster 8 Tage", "Feste Frist 8 Tage"):
        assert _analysis(name).km.status == "ok", name
    # „rund jeder fünfte“ beim Fenster 30 Tage, „gut die Hälfte“ beim Fenster 8 Tage, „deutlich weniger“ bei der festen Frist 8 Tage
    assert 0.15 <= _analysis("Fenster 30 Tage").censored_share <= 0.25
    assert 0.5 < _analysis("Fenster 8 Tage").censored_share < 0.65
    assert _analysis("Feste Frist 8 Tage").censored_share < _analysis("Fenster 8 Tage").censored_share


def test_preset_fenster_4_tage_hilfe_nennt_die_gemessene_haelfte():
    """„in der Messung gilt das für gut die Hälfte der Zufalls-Seeds“: Anteil ohne Median bei Fenster 4 Tage, n = 300, steht in der eingefrorenen Messung."""
    row = next(r for r in ev.measurements()["defined"] if r["length"] == 4)
    assert 0.5 < row["undef_km"] < 0.65


def test_settings_haben_dieselben_grenzen_wie_die_regler():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("window_slider") == (C.W_MIN, C.W_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
