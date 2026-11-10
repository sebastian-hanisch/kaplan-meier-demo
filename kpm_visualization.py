"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Zahlen in Beschriftungen werden gerundet (Übertragung klein halten); Zensierungsmarken sind auf höchstens MAX_MARKS ausgedünnt."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kpm_algorithm as A
import kpm_constants as C
import kpm_core as K
import kpm_evaluation as E
import kpm_scenario as S

MAX_MARKS = 120
X_CAP = 30.0   # Achse höchstens bis Tag 30: dahinter liegt kaum noch etwas


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def survival_values(km: dict, x) -> np.ndarray:
    """Ŝ an den Stellen x (vektorisiert; vor der ersten Ereigniszeit 1)."""
    x = np.asarray(x, dtype=float)
    if km["t"].size == 0:
        return np.ones_like(x)
    idx = np.searchsorted(km["t"], x, side="right") - 1
    return np.where(idx < 0, 1.0, km["s"][np.maximum(idx, 0)])


def step_xy(km: dict, x_end: float):
    """Treppe als Punktliste für line_shape="hv": von (0, 1) über alle Ereigniszeiten bis x_end."""
    xs = np.concatenate([[0.0], km["t"], [x_end]])
    last = km["s"][-1] if km["s"].size else 1.0
    ys = np.concatenate([[1.0], km["s"], [last]])
    return np.round(xs, 4), np.round(ys, 5)


def x_end(sample: S.Sample) -> float:
    return float(min(sample.obs.max(), X_CAP))


def _thin(values: np.ndarray) -> np.ndarray:
    if values.size <= MAX_MARKS:
        return values
    return values[np.linspace(0, values.size - 1, MAX_MARKS).astype(int)]


def _censor_marks(sample: S.Sample, km: dict, xmax: float):
    t = np.sort(sample.obs[~sample.event])
    t = _thin(t[t <= xmax])
    return np.round(t, 4), np.round(survival_values(km, t), 5)


def _truth_trace(xmax: float):
    grid = np.linspace(0.0, xmax, 160)
    return go.Scatter(x=np.round(grid, 3), y=np.round(S.true_survival(grid), 5), mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"),
                      name="Wahrheit (lognormal)", hovertemplate="t = %{x:.1f}: S = %{y:.3f}<extra>Wahrheit</extra>")


def _curve_trace(km: dict, xmax: float, name: str, color: str, width: int = 3):
    xs, ys = step_xy(km, xmax)
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, shape="hv"), name=name,
                      hovertemplate="t = %{x:.2f}: S = %{y:.3f}<extra>" + name + "</extra>")


def _axes_survival(fig, xmax: float, **kw):
    fig.update_xaxes(title_text="Tage seit Ankunft", range=[0, xmax * 1.02], **kw)
    fig.update_yaxes(title_text="Anteil noch da, S(t)", range=[0, 1.02], **kw)


def _staircase(rows: list[dict], k: int, x_end: float):
    """Treppe der ersten k Ereigniszeiten des Handbeispiels als Punktliste (senkrechte Sprünge), bis x_end verlängert."""
    xs, ys, prev = [0.0], [1.0], 1.0
    for r in rows[:k]:
        xs += [float(r["t"]), float(r["t"])]
        ys += [prev, float(r["s"])]
        prev = float(r["s"])
    xs.append(float(x_end))
    ys.append(prev)
    return xs, ys


def build_hand(rows: list[dict], step: int, times=S.HAND_TIMES, events=S.HAND_EVENTS):
    """Handbeispiel: oben die sechs Lebenslinien (● Abholung, + noch da), unten die Kurve, die mit jeder verarbeiteten Ereigniszeit wächst."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.42, 0.58], vertical_spacing=0.08)
    ids = list(range(len(times), 0, -1))
    for i, (t, e) in enumerate(zip(times, events)):
        fig.add_trace(go.Scatter(x=[0, t], y=[ids[i]] * 2, mode="lines", line=dict(color="rgba(150,150,150,0.8)", width=3), showlegend=False, hoverinfo="skip"), row=1, col=1)
    done = [t for t, e in zip(times, events) if e]
    fig.add_trace(go.Scatter(x=done, y=[ids[i] for i, e in enumerate(events) if e], mode="markers", marker=dict(symbol="circle", size=11, color=C.COLOR_KM),
                             name="Abholung gesehen", hovertemplate="Container %{y}: abgeholt nach %{x} Tagen<extra></extra>"), row=1, col=1)
    cens = [t for t, e in zip(times, events) if not e]
    fig.add_trace(go.Scatter(x=cens, y=[ids[i] for i, e in enumerate(events) if not e], mode="markers", marker=dict(symbol="cross", size=12, color=C.COLOR_ALL, line=dict(width=2)),
                             name="noch da (zensiert)", hovertemplate="Container %{y}: noch da nach %{x} Tagen<extra></extra>"), row=1, col=1)
    tmax = float(max(times))
    full_x, full_y = _staircase(rows, len(rows), tmax)
    fig.add_trace(go.Scatter(x=full_x, y=full_y, mode="lines", line=dict(color="rgba(150,150,150,0.55)", width=2, dash="dot"), name="ganze Kurve", hoverinfo="skip"), row=2, col=1)
    reach = float(rows[step]["t"]) if step < len(rows) else tmax
    xs, ys = _staircase(rows, step, reach)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_KM, width=4), name="bisher berechnet", hovertemplate="t = %{x}: S = %{y:.4f}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=0.5, line=dict(color=C.COLOR_TRUTH, width=1, dash="dash"), row=2, col=1)
    fig.update_xaxes(range=[0, tmax + 0.3], row=2, col=1, title_text="Tage")
    fig.update_yaxes(title_text="Container", tickmode="array", tickvals=ids, ticktext=[str(i) for i in range(1, len(times) + 1)], range=[0.4, len(times) + 0.6], row=1, col=1)
    fig.update_yaxes(title_text="S(t)", range=[0, 1.05], row=2, col=1)
    return _base(fig, height=470)


def build_ways(a: E.Analysis):
    """Die Kurven der drei Wege gegen die Wahrheit; die Zensierungsmarken liegen auf der Kaplan–Meier-Kurve."""
    xmax = x_end(a.sample)
    fig = go.Figure()
    fig.add_trace(_truth_trace(xmax))
    for name, color, width in ((A.WAY_EVENTS, C.COLOR_EVENTS, 2), (A.WAY_ALL, C.COLOR_ALL, 2)):
        km = a.ways[name].km
        if km["t"].size:
            fig.add_trace(_curve_trace(km, xmax, name, color, width))
    kmw = a.km.km
    if kmw["t"].size:
        fig.add_trace(_curve_trace(kmw, xmax, A.WAY_KM, C.COLOR_KM, 3))
        cx, cy = _censor_marks(a.sample, kmw, xmax)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", marker=dict(symbol="line-ns-open", size=9, color=C.COLOR_KM), name="noch da (zensiert)",
                                 hovertemplate="noch da nach %{x:.2f} Tagen<extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color="rgba(120,120,120,0.6)", width=1, dash="dash"))
    _axes_survival(fig, xmax)
    return _base(fig)


def build_band(a: E.Analysis, method: str):
    """Kaplan–Meier mit punktweisem 95-%-Band (log-log oder einfach) und der wahren Kurve."""
    xmax = x_end(a.sample)
    km = a.km.km
    fig = go.Figure()
    if km["t"].size:
        lo, hi = K.band_curve(km, C.LEVEL, method)
        xs = np.round(np.concatenate([[0.0], km["t"], [xmax]]), 4)
        lo_y = np.concatenate([[1.0], lo, [lo[-1]]])
        hi_y = np.concatenate([[1.0], hi, [hi[-1]]])
        color = C.COLOR_BAND if method == "loglog" else C.COLOR_BAND_PLAIN
        fig.add_trace(go.Scatter(x=xs, y=np.round(hi_y, 5), mode="lines", line=dict(width=0, shape="hv"), showlegend=False, hoverinfo="skip", connectgaps=False))
        fig.add_trace(go.Scatter(x=xs, y=np.round(lo_y, 5), mode="lines", line=dict(width=0, shape="hv"), fill="tonexty", fillcolor=color, name="95-%-Band (nominal)", hoverinfo="skip",
                                 connectgaps=False))
        fig.add_trace(_curve_trace(km, xmax, A.WAY_KM, C.COLOR_KM, 3))
    fig.add_trace(_truth_trace(xmax))
    for row in E.band_checks(a, method):
        if row["inside"] is None:
            continue
        fig.add_trace(go.Scatter(x=[row["t"]], y=[row["truth"]], mode="markers", showlegend=False,
                                 marker=dict(symbol="circle" if row["inside"] else "x", size=11, color=C.COLOR_KM if row["inside"] else C.COLOR_ALL, line=dict(width=2, color="white")),
                                 hovertemplate=f"t = {row['t']}: wahres S = {row['truth']:.3f}, " + ("im Band" if row["inside"] else "außerhalb") + "<extra></extra>"))
    _axes_survival(fig, xmax)
    return _base(fig)


def build_defined(rows: list[dict], current_window: int | None):
    """Anteil der Läufe, in denen der Kaplan–Meier-Median schätzbar ist, je Fensterlänge (eingefrorene Messung)."""
    xs = [str(r["length"]) for r in rows]
    ys = [round(100.0 * (1.0 - r["undef_km"]), 1) for r in rows]
    colors = [C.COLOR_KM if current_window is not None and r["length"] == current_window else C.COLOR_LINE for r in rows]
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=colors, text=[E.de(y, 1) + " %" for y in ys], textposition="outside", hovertemplate="Fenster %{x} Tage: %{y:.1f} % der Läufe<extra></extra>"))
    fig.update_xaxes(title_text="Länge des Beobachtungsfensters (Tage)", type="category")
    fig.update_yaxes(title_text="Läufe mit schätzbarem Median (%)", range=[0, 112])
    return _base(fig, height=320)


def build_rmst(a: E.Analysis, tau: float):
    """Kaplan–Meier bis τ; die Fläche unter der Treppe ist die mittlere Lebenszeit bis τ."""
    km = a.km.km
    fig = go.Figure()
    xs, ys = step_xy(km, float(tau))
    keep = xs <= tau + 1e-9
    fig.add_trace(go.Scatter(x=xs[keep], y=ys[keep], mode="lines", line=dict(color=C.COLOR_KM, width=3, shape="hv"), fill="tozeroy", fillcolor=C.COLOR_BAND, name="Kaplan–Meier bis τ",
                             hovertemplate="t = %{x:.2f}: S = %{y:.3f}<extra></extra>"))
    grid = np.linspace(0.0, float(tau), 120)
    fig.add_trace(go.Scatter(x=np.round(grid, 3), y=np.round(S.true_survival(grid), 5), mode="lines", line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), name="Wahrheit"))
    _axes_survival(fig, float(tau))
    return _base(fig, height=320)
