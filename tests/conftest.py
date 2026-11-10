"""Gemeinsame Fixtures: das Handbeispiel (sechs Container), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def six():
    """Sechs Container, Zeiten 2, 3+, 4, 5, 5+, 8 (+ = beim Ende der Beobachtung noch da).
    Handrechnung: t = 2: n = 6, d = 1, S = 5/6; t = 4: n = 4, d = 1, S = 5/8; t = 5: n = 3 (der Zensierte bei 5 bleibt im Risiko), d = 1, S = 5/12; t = 8: n = 1, d = 1, S = 0.
    Median 5; Mittel aller Zeiten 9/2; Mittel nur der Ereignisse 19/4; mittlere Lebenszeit bis 8: 133/24."""
    return [2, 3, 4, 5, 5, 8], [1, 0, 1, 1, 0, 1]
