# Kaplan–Meier: zensierte Zeiten lesen – Streamlit-Demo

<!-- Live-Link nach dem Deploy direkt hierher setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kaplan-meier-demo.streamlit.app/)** -->

Wurzel der Reihe **Überlebenszeit** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Interaktive Demo zu einer Frage, die jede Beobachtung von Zeiten bis zu einem Ereignis stellt:
**Was fängt man mit Fällen an, die bei Ende der Beobachtung noch laufen?** Vehikel ist die **Containerverweilzeit** der [verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo) (lognormal, Median 4 Tage, σ = 0,9; dort ohne Zensierung):
Ein Container kommt an und wird irgendwann abgeholt; die Beobachtung endet an einem Stichtag, und wer dann noch im Block steht, ist **zensiert**. Die Daten sind erfunden und die Wahrheit ist bekannt, so lässt sich jeder Weg gegen sie messen.
Die Zensierungsarten (Beobachtungsfenster und feste Frist) und die naiven Auswege (Mittel aller beobachteten Zeiten, nur Ereignisse, Zensierte als Ereignis) sind Regler und Auswahlen dieser einen Demo.

**Annahme, die in jeder Stufe der Demo gilt und sichtbar bleibt:** Die Zensierung ist unabhängig vom Ereignis. Ob ein Container beim Ende noch da ist, hängt hier nur davon ab, wann er ankam (Fenster) bzw. von der festen Frist, nicht davon, wie bald er abgeholt worden wäre.
Das gilt, weil die Daten so gebaut sind; in echten Daten ist es eine Annahme, die Kaplan–Meier nicht prüft.

## Warum dieses Problem

Eine Verweilzeit, die beim Ende der Beobachtung noch läuft, ist keine kurze Verweilzeit, sondern eine, von der man nur „mindestens so lang“ weiß. Das **Mittel aller beobachteten Zeiten** und das **Weglassen der noch anwesenden Container** behandeln sie falsch;
**Kaplan–Meier** benutzt jeden Container genau bis zu dem Tag, an dem seine Beobachtung endet. Kernfrage der Demo: **Wie weit liegen die Wege daneben, und was kann Kaplan–Meier nicht?**

Gemessen in der erfundenen Welt (Beobachtungsfenster von 30 Tagen, n = 300, 19,6 % zensiert, 1000 Läufe): Das Mittel aller beobachteten Zeiten liegt bei 4,73 ± 0,01 Tagen gegen das wahre Mittel 6,00 (−21,1 % gegen die Wahrheit), der Median von Kaplan–Meier bei 4,00 ± 0,01 gegen den wahren Median 4,00.
Bei einem Fenster von 8 Tagen (55,6 % zensiert) sinkt das Mittel auf 2,80 (−53,3 %), der Median von Kaplan–Meier bleibt bei 4,00. **Grenze:** Bei einem Fenster von 4 Tagen (= wahrer Median, n = 300, 77,6 % zensiert) hat Kaplan–Meier in 55,4 % der Läufe keinen Median, weil die Kurve in diesen Läufen 0,5 nicht erreicht.

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (Phase 0, Vormessung mit numpy-Zufall) hat sich beim Bau an acht Stellen geändert:

1. **Statt vier Wegen gibt es drei.** Der Plan nennt „Mittel aller“, „nur Ereignisse“, „Zensierte als Ereignis“ und Kaplan–Meier. „Mittel aller beobachteten Zeiten“ und „Zensierte als Ereignis“ sind derselbe Weg: Wer jede beobachtete Zeit als Abholung zu diesem Zeitpunkt liest, hat die Kurve 1 − empirische Verteilungsfunktion, deren Mittel das Mittel aller Zeiten ist und deren Median der untere Median aller Zeiten (kleinste Zeit mit Ŝ ≤ 0,5; bei geradem n etwas tiefer als der übliche Median, bei n = 300 um rund 0,015 Tage nach einer Prüfrechnung) (`tests/test_algorithm.py`, `tests/test_oracle_kaplan_meier.py`).
2. **Median gegen Median.** Die Vormessung verglich das Mittel (Wahrheit 6,00) mit dem Kaplan–Meier-Median (Wahrheit 4,00). Die Demo stellt zusätzlich die Mediane der naiven Wege daneben: 3,43 ± 0,01 („Zensierte als Ereignis“) und 3,41 ± 0,01 („nur Ereignisse“) bei Fenster 30 Tage, n = 300 (−14,3 % und −14,7 % gegen den wahren Median).
3. **Eigene Zahlen, 1000 statt 200 Läufe.** Kern und Zufall (SplitMix64) sind neu; die Zahlen weichen zufallsbedingt von der Vormessung ab (`vormessung/out_uz_v1_bis_v8.txt`: Mittel 4,76, Kaplan–Meier-Median 4,03, bei Fenster 4 Tage 54 % ohne Median; hier 4,73, 4,00 und 55,4 %). Es gelten die Zahlen dieser README (`tools/messung_kaplan_meier.json`).
4. **„Median nur in 46 % der Läufe definiert“ hängt an der Fensterlänge; bei Fenster = wahrer Median (4 Tage) steigt der Anteil mit n kaum.** Fenster 4 Tage: 42,2 % (n = 100), 44,6 % (n = 300), 45,0 % (n = 1000) der Läufe mit Median (Binomial-Standardfehler je Wert rund 1,6 Prozentpunkte): kein Hinweis, dass mehr Container den Median bei diesem Fenster schätzbar machen. Liegt das Fenster über dem wahren Median (5 Tage), wächst der Anteil nach einer Prüfrechnung mit n gegen 100 % (65 % bei n = 50, 79 % bei n = 100, 94 % bei n = 300, 100 % bei n = 1000), liegt es darunter (3 Tage), fällt er mit n gegen 0. Der Mittelwert der Mediane der Läufe, in denen es einen gibt, steigt dagegen mit n (3,40, 3,61, 3,76): Er gilt nur für diese Läufe und ist kein Schätzwert für den Median.
5. **Greenwood-Band: gemessene Überdeckung 93,6 bis 95,9 % (log-log) und 90,1 bis 95,2 % (einfach)** über elf Zellen mit je 1000 Läufen. Der Plan nennt 93 bis 97 % und für den einfachen Streifen bei n = 100, t = 12 den Ausreißer 90,0 %; er bleibt (hier 90,1 %, zwei der 1000 Läufe haben dort kein Band). „Nominal 95 %“ steht ohne die Spanne des Plans.
6. **Die Eigenschaft „eine Zensierung hinter dem letzten Ereignis ändert die Kurve nicht“ gilt nur für die Stufenzeiten.** Der Zensierte bleibt in jeder Risikomenge: Jeder Faktor wird (n + 1 − d)/(n + 1), die Werte der Kurve steigen (`tests/test_eigenschaften_kaplan_meier.py`).
7. **Typ II (Stopp nach r Abholungen) ist nicht umgesetzt.** Die Demo zeigt Beobachtungsfenster und feste Frist (Typ I).
8. **Orakel ohne Sub-Agent.** Das Orakel ist ein unabhängig geschriebenes Umverteilungsverfahren in Brüchen, kein vom Plan vorgesehener fremder Rechenweg eines Sub-Agenten; der Fremdvergleich mit `lifelines` fehlt (nicht im Test-venv).

Zusatzbefund: Bei der **festen Frist** ist der Median von „Zensierte als Ereignis“ richtig, solange die Frist über dem wahren Median liegt (3,99 ± 0,01 bei Frist 30 und 8 Tage, wahr 4,00; untere Median-Konvention, beide Fristen nutzen dieselben Zufalls-Seeds und sind keine zwei unabhängigen Bestätigungen), das Mittel nicht (5,83 und 4,45, wahr 6,00). Grund (Rechnung, keine Messung): Mehr als die Hälfte der Container wird vor der Frist abgeholt, der Median von min(D, Frist) ist dann der Median von D.

## Modell

Verweilzeit D lognormal mit Median 4 Tagen und σ = 0,9 (wahres Mittel 5,997, wahres 0,9-Quantil 12,68; geschlossene Formeln in `kpm_scenario.py`). Beobachtet wird T = min(D, C), das Ereignis ist D ≤ C. Zwei Arten der Zensierung C:
**Beobachtungsfenster** der Länge W (Ankunft gleichverteilt in den ersten W Tagen, die Beobachtung endet für alle am Tag W, also C = W − Ankunft) und **feste Frist** (jeder Container wird W Tage beobachtet, C = W). Die Formeln stehen im Expander „📐 Mathematische Formulierung“ der App.
Alle Zufallszahlen der Stichproben kommen aus SplitMix64 (`kpm_core.py`), nicht aus `hash()` oder numpy-Verteilungen; der Knopf „Neue Stichprobe“ wählt den Zufalls-Seed mit Pythons `random`. Zufalls-Seed und Einstellungen bestimmen die Stichprobe vollständig.

## Methodik

- **Zensierte als Ereignis** (= Mittel aller beobachteten Zeiten): jede beobachtete Zeit zählt als Abholung. **Nur Ereignisse**: Zensierte fehlen. **Kaplan–Meier**: Produkt der Faktoren (n − d)/n an den Ereigniszeiten, Zensierte bleiben bis zu ihrem Beobachtungsende im Risiko. Bei gleicher Zeit zählt das Ereignis vor der Zensierung, der Zensierte bleibt im Risiko.
- **Median:** kleinste Ereigniszeit mit Ŝ ≤ 0,5; erreicht die Kurve 0,5 nicht, meldet die App „kein Median“ statt einer Zahl.
- **Greenwood-Band** (punktweise, nominal 95 %): einfach (Ŝ ± z·Standardfehler, auf [0, 1] gekappt) und log-log.
- **Mittlere Lebenszeit bis τ:** Fläche unter der Treppe; τ über der größten Beobachtungszeit meldet die App als nicht schätzbar.
- **Die vier Experimente:** (1) das Handbeispiel mit sechs Containern in Brüchen, (2) die Wege über 30 feste Zufalls-Seeds und die eingefrorene Messung, (3) das Greenwood-Band mit Überdeckungstabelle, (4) „Median nicht schätzbar“ mit Quantilen und mittlerer Lebenszeit.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Messung `tools/measure_kaplan_meier.py` → `tools/messung_kaplan_meier.json` (Welt wie oben, 1000 Läufe je Zeile, Zufalls-Seeds 2000 bis 2999; Überdeckung 5000 bis 5999, mittlere Lebenszeit 6000 bis 6999), Mittel über die Läufe ± Standardfehler des Mittels.
Die Spalte „Prüfung“ nennt die Tests, die die Zahl gegen die Messdatei und diese README halten (`tests/test_claims.py`; die Neuberechnung ausgewählter Zellen vergleicht mit Toleranz, nicht exakt).

| Frage (Beobachtungsfenster, n = 300, wenn nicht anders genannt) | Befund | Prüfung |
|---|---|---|
| Fenster 30 Tage (19,6 % zensiert): Mittel aller Zeiten, nur Ereignisse, Median Kaplan–Meier | 4,73 ± 0,01 (−21,1 %), 4,53 ± 0,01 (−24,5 %), 4,00 ± 0,01 (wahr: Mittel 6,00, Median 4,00) | `test_messung_gegen_neu_berechnung`, `test_readme_zahlen` |
| Fenster 8 Tage (55,6 % zensiert): dieselben drei | 2,80 ± 0,00 (−53,3 %), 2,60 ± 0,00 (−56,6 %), 4,00 ± 0,01 | `test_messung_gegen_neu_berechnung`, `test_readme_zahlen` |
| Fenster 4 Tage (77,6 % zensiert): Mittel aller, Median Kaplan–Meier | 1,74 ± 0,00 (−71,0 %); Median in 44,6 % der Läufe vorhanden, dort 3,61 ± 0,01 | `test_messung_gegen_neu_berechnung`, `test_readme_zahlen` |
| Mittel aller bei n = 100, 300, 1000 (Fenster 30 Tage) | 4,74 ± 0,01, 4,73 ± 0,01, 4,74 ± 0,00: mehr Daten reparieren das Mittel nicht (−21,0 %, −21,1 %, −21,0 %) | `test_readme_zahlen` |
| Median der naiven Wege bei Fenster 30, 8, 4 Tage | „Zensierte als Ereignis“ 3,43, 2,39, 1,62; „nur Ereignisse“ 3,41, 2,30, 1,58 (wahr 4,00) | `test_readme_zahlen` |
| Feste Frist 30, 8, 4 Tage (1,3 %, 22,1 %, 49,9 % zensiert): Median „Zensierte als Ereignis“ | 3,99 ± 0,01, 3,99 ± 0,01, 3,89 ± 0,00; Median „nur Ereignisse“ 3,94, 3,10, 2,17; Mittel aller 5,83, 4,45, 3,10 | `test_readme_zahlen` |
| Median schätzbar bei Fenster 4, 5, 6, 8, 12, 30 Tage | 44,6 %, 90,7 %, 99,8 %, 100,0 %, 100,0 %, 100,0 % der Läufe | `test_readme_zahlen`, `test_preset_fenster_4_tage_hilfe_nennt_die_gemessene_haelfte` |
| Überdeckung des nominalen 95-%-Bandes (11 Zellen, je 1000 Läufe) | log-log 93,6 bis 95,9 %; einfach 90,1 bis 95,2 %, am kleinsten bei Fenster 30 Tage, n = 100, t = 12 | `test_readme_zahlen`, `test_ueberdeckung_neu_berechnet` |
| Mittlere Lebenszeit bis τ, Fenster 8 Tage, τ = 6 (wahr 3,915) | Kaplan–Meier 3,92 ± 0,00 (+0,2 %); Mittel von min(Zeit, τ) 2,73 ± 0,00 (−30,2 %) | `test_readme_zahlen`, `test_mittlere_lebenszeit_neu_berechnet` |
| Mittlere Lebenszeit bis τ, Fenster 30 Tage, τ = 12 (wahr 5,086) | Kaplan–Meier 5,10 ± 0,01 (+0,3 %); Mittel von min(Zeit, τ) 4,45 ± 0,01 (−12,5 %) | `test_readme_zahlen`, `test_mittlere_lebenszeit_neu_berechnet` |
| Handbeispiel 2, 3+, 4, 5, 5+, 8 | S = 5/6, 5/8, 5/12, 0; Median 5; Mittel aller 9/2, nur Ereignisse 19/4; mittlere Lebenszeit bis 8: 133/24; Greenwood-Varianz bei t = 5: 85/1728 | `tests/test_algorithm.py`, `tests/test_core.py` |

Die App rechnet zusätzlich live über 30 feste Zufalls-Seeds (7000 bis 7029) für die gewählten Einstellungen; diese Zahlen stehen in keiner README, nur in der App.

## Ehrliche Grenzen

- **Eine Welt:** lognormal, Median 4, σ = 0,9, zwei Zensierungsarten, beide unabhängig vom Ereignis. Andere Verteilungen, Typ II, Linkszensierung und Intervallzensierung sind nicht gemessen.
- **Die Annahme „Zensierung unabhängig“ wird nicht geprüft.** Wie weit Kaplan–Meier danebenliegt, wenn sie verletzt ist, zeigt diese Demo nicht; die Richtung folgt aus der Rechnung (Container, die kurz vor der Abholung ausscheiden, blieben als „noch da“ in der Risikomenge), die Größe ist nicht gemessen.
- **Bänder punktweise**, nicht als Streifen über die ganze Kurve; Überdeckung nur an den elf Zellen der Tabelle gemessen. Der einfache Streifen liegt bei n = 100 und t = 12 bei 90,1 %, das log-log-Band dort bei 94,2 %.
- **Eigener Kern:** Kaplan–Meier ist selbst geschrieben und gegen ein Umverteilungsverfahren in Brüchen, eine Bruchrechnung des Greenwood-Terms, einen Bootstrap und `scipy` für die Wahrheit geprüft; ein Fremdvergleich mit `lifelines` oder `statsmodels` fehlt.
- **Quellen nicht eingesehen:** Kaplan und Meier (1958) und Klein/Moeschberger (Survival Analysis) stehen nur als Hinweis (aus dem Gedächtnis bzw. Suchtreffer); kein Wert der Demo stammt aus ihnen.
- **Kaplan–Meier hat in der Regel kein Mittel der Verweilzeit:** Ist die größte beobachtete Zeit zensiert, endet die Kurve oberhalb von null (nach einer Prüfrechnung bei n = 300 und Fenster 30 Tage in rund 71 % der Läufe, bei Fenster 8 und 4 Tagen in mehr als 97 %); die App meldet das Mittel immer als „nicht definiert“. Die mittlere Lebenszeit bis τ ist ein anderes Maß.
- **Darstellung:** Die Diagramme sind im Browser angesehen (hell, Breite 1280), nicht auf Telefon und nicht im dunklen Design; Mausklicks in den Diagrammen lösen nichts aus.

## Tests

120 Tests (`pytest tests/`, Laufzeit allein rund 20 Sekunden, bei hoher Rechnerlast einige Minuten), davon:

- **Kern** (`tests/test_core.py`): Handbeispiel mit Brüchen, Gleichstand, Greenwood-Band (Handformel, einfach und log-log), Eingabeprüfung, SplitMix64 (Referenzwert von Vigna).
- **Orakel** (`tests/test_oracle_kaplan_meier.py`): Kaplan–Meier gegen das Umverteilungsverfahren nach rechts in Brüchen (120 Zufallsinstanzen mit Gleichständen), ohne Zensierung gegen Aufzählung, Greenwood gegen Bruchrechnung (60 Instanzen) und gegen Bootstrap (5 Stichproben), mittlere Lebenszeit gegen Fläche in Brüchen und Gitter-Quadratur, Wahrheit gegen `scipy.stats.lognorm`, Unverzerrtheit an einer Welt mit numpy-Zufall (200 Läufe).
- **Eigenschaften** (`tests/test_eigenschaften_kaplan_meier.py`): Monotonie, Umsortieren, Skalierung, Verschiebung, Zensierung hinter dem letzten und vor dem ersten Ereignis, Ereignis zu Zensierung, Band wird mit dem Niveau breiter und mit n schmaler.
- **Wege, Szenario, Randfälle, Verdikte** (`test_algorithm.py`, `test_scenario.py`, `test_edge_cases.py`, `test_verdicts.py`): jede Einheit an der Mini-Instanz (sechs Container), Status statt stiller Zahlen (`keine_ereignisse`, `nicht_schaetzbar`, `tau_ueber_beobachtung`), jedes Verdikt mit jeder Zahl im Text.
- **Aussagen** (`tests/test_claims.py`): README-Zahlen gegen die eingefrorene Messung, Neuberechnung ausgewählter Zellen, Messdatei gegen Neuberechnung.
- **Oberfläche** (`test_app.py`, `test_layout.py`, `test_presets.py`, `test_pdf.py`, `test_footer.py`, `test_runtime_imports.py`): Rauchtest jedes Buttons und Reglers an den Grenzen, alle Minima und alle Maxima zugleich, Schnellstart in der Seitenleiste mit Hervorhebung, Footer, Laufzeit-Importe gegen `requirements.txt`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `kpm_algorithm.py` | die drei Wege (Zensierte als Ereignis, nur Ereignisse, Kaplan–Meier) und mittlere Lebenszeit mit Status |
| `kpm_constants.py` | Regler-Grenzen, Welt, Farben, Presets, feste Zufalls-Seeds |
| `kpm_core.py` | Kern (Kopie aus ueberlebenszeit-planung/kern/uz_core.py, Stand 2026-11-10, gekürzt): Kaplan–Meier, Greenwood-Band, mittlere Lebenszeit, SplitMix64 |
| `kpm_evaluation.py` | Analyse, Läufe über feste Zufalls-Seeds, Handrechnung in Brüchen, Verdikte als reine Funktionen |
| `kpm_layout.py` | Fester Kopf (Stylesheet) |
| `kpm_pdf_export.py` | PDF-Zusammenfassung (fpdf2, erst nach Klick) |
| `kpm_presets.py` | Permalink (Begrenzen), Schnellstart in der Seitenleiste mit Hervorhebung, Zufalls-Seed-Button |
| `kpm_scenario.py` | Erzeuger der Stichproben (Fenster, feste Frist) und die Wahrheit in geschlossener Form |
| `kpm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tools/measure_kaplan_meier.py` | Messskript für alle Zahlen dieser README |
| `tools/messung_kaplan_meier.json` | eingefrorenes Ergebnis der Messung (die App liest daraus die Tabellen) |
| `tests/` | Tests, siehe oben |

## Bewusst nicht umgesetzt

- **Typ II** (Stopp nach r Abholungen), **Linkszensierung** (verzögerter Eintritt), **Intervallzensierung**, **konkurrierende Abholwege**: andere Fragen, nicht in dieser Demo.
- **Vergleich zweier Gruppen, Merkmale, parametrische Familien:** Kaplan–Meier liest eine Kurve für alle Container gemeinsam.
- **Abhängige (informative) Zensierung:** Sie wird als Annahme benannt und nicht gemessen.
- **Streifenbänder** über die ganze Kurve und Bootstrap-Bänder in der App.
- **Fremdpaket-Vergleich** mit `lifelines` oder `statsmodels`.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die in der Streamlit Cloud eingestellte Python-Version ist nicht geprüft; die CI läuft mit Python 3.12. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Messung neu erzeugen: `python tools/measure_kaplan_meier.py` (schreibt `tools/messung_kaplan_meier.json`).

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
