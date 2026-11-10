"""Konstanten: Regler-Grenzen, feste Annahmen der Welt, Farben und Presets."""

# --- Die Welt: Verweilzeit D eines Containers im Block, lognormal (wie in der Verweilzeit-Demo) ------------------------------------------------
MEDIAN_TRUE = 4.0            # Tage
SIGMA_TRUE = 0.9             # σ der Log-Skala; ergibt Mittel 4·exp(0,405) = 5,997 und das 0,9-Quantil 12,7

# --- Zensierungsarten (Auswahl in der Seitenleiste) ---------------------------------------------------------------------------------------------
KIND_WINDOW = "Beobachtungsfenster (Ankunft zufällig)"
KIND_FIXED = "Feste Frist je Container (Typ I)"
KINDS = (KIND_WINDOW, KIND_FIXED)

# --- Regler -------------------------------------------------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 20, 1000, 300            # Kaplan–Meier ist O(n log n); der Median über 30 feste Seeds braucht bei n = 1000 einige Zehntelsekunden
W_MIN, W_MAX, DEFAULT_W = 2, 60, 30                # Länge des Fensters bzw. der Frist in Tagen
SEED_MAX, DEFAULT_SEED = 999999, 2026
TAU_MIN, TAU_MAX, DEFAULT_TAU = 1, 60, 8           # τ der mittleren Lebenszeit (Experiment 4)

# --- Messungen ----------------------------------------------------------------------------------------------------------------------------------
BAND_SEEDS = tuple(range(7000, 7030))              # 30 feste Zufalls-Seeds für das Band der Experimente (nie ein einzelner Seed als Befund)
LEVEL = 0.95                                       # nominales Niveau des Greenwood-Bandes
MEASURE_FILE = "tools/messung_kaplan_meier.json"   # eingefrorenes Ergebnis von tools/measure_kaplan_meier.py (1000 Läufe je Zelle)

# --- Farben -------------------------------------------------------------------------------------------------------------------------------------
COLOR_TRUTH = "#555555"
COLOR_KM = "#2F6B65"
COLOR_ALL = "#d1495b"        # „Zensierte als Ereignis“
COLOR_EVENTS = "#edae49"     # „nur Ereignisse“
COLOR_BAND = "rgba(47,107,101,0.20)"
COLOR_BAND_PLAIN = "rgba(76,120,168,0.20)"
COLOR_LINE = "#4c78a8"

PRESETS = {
    "Fenster 30 Tage": {"n": DEFAULT_N, "window": 30, "kind": KIND_WINDOW, "seed": DEFAULT_SEED},
    "Fenster 8 Tage": {"n": DEFAULT_N, "window": 8, "kind": KIND_WINDOW, "seed": DEFAULT_SEED},
    "Fenster 4 Tage": {"n": DEFAULT_N, "window": 4, "kind": KIND_WINDOW, "seed": 1},
    "Feste Frist 8 Tage": {"n": DEFAULT_N, "window": 8, "kind": KIND_FIXED, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Fenster 30 Tage": "300 Container, die Beobachtung endet nach 30 Tagen, die Ankunft ist gleichverteilt: rund jeder fünfte Container ist beim Ende noch da. Die Abweichungen der Wege stehen unter „Ergebnis“.",
    "Fenster 8 Tage": "Dasselbe mit einem Fenster von 8 Tagen: gut die Hälfte der Container ist beim Ende noch da.",
    "Fenster 4 Tage": "Fenster von 4 Tagen: mehr als drei Viertel der Container sind zensiert. Bei diesem Zufalls-Seed ist der Median von Kaplan–Meier nicht schätzbar (in der Messung gilt das für gut die Hälfte der Zufalls-Seeds).",
    "Feste Frist 8 Tage": "Jeder Container wird genau 8 Tage lang beobachtet (Typ I): weniger Zensierung als beim Fenster gleicher Länge.",
}
