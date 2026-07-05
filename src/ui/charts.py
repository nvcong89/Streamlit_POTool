"""Plotly charts for stress-strain, M-φ, and pushover hinge."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis.pushover_derivation import HingeBackbone


MATERIAL_COLORS = {
    "unconfined_concrete": "#6b7280",
    "confined_concrete": "#374151",
    "rebar_steel": "#2563eb",
    "bearing_steel": "#dc2626",
}


def plot_stress_strain(material_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for material in material_df["material"].unique():
        sub = material_df[material_df["material"] == material]
        fig.add_trace(
            go.Scatter(
                x=sub["strain"],
                y=sub["stress"],
                mode="lines",
                name=material.replace("_", " ").title(),
                line=dict(color=MATERIAL_COLORS.get(material, "#111827"), width=2),
            )
        )
    fig.update_layout(
        title="Stress–Strain Curves",
        xaxis_title="Strain",
        yaxis_title="Stress (MPa)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=60, b=50),
    )
    return fig


def plot_mc_family(mc_df: pd.DataFrame, selected_P: Optional[float] = None) -> go.Figure:
    fig = go.Figure()
    unique_P = sorted(mc_df["P"].unique())
    for P_val in unique_P:
        sub = mc_df[mc_df["P"] == P_val].sort_values("curvature")
        sub = sub[sub["curvature"] >= 0]
        width = 3 if selected_P is not None and abs(P_val - selected_P) < 1.0 else 1
        opacity = 1.0 if selected_P is not None and abs(P_val - selected_P) < 1.0 else 0.45
        fig.add_trace(
            go.Scatter(
                x=sub["curvature"],
                y=sub["M"],
                mode="lines",
                name=f"P = {P_val:.0f} kN",
                line=dict(width=width),
                opacity=opacity,
            )
        )
    fig.update_layout(
        title="Moment–Curvature (M–φ)",
        xaxis_title="Curvature (1/mm × 1000)",
        yaxis_title="Moment M (kNm)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_pm_curve(pm_df: pd.DataFrame, mirror: bool = True) -> go.Figure:
    """P (kN) vs Mp (kNm) interaction envelope."""
    env = pm_df.sort_values("P")
    pos = env[env["Mp"] > 0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pos["P"],
            y=pos["Mp"],
            mode="lines+markers",
            name="P–M envelope",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=6),
        )
    )
    if mirror:
        fig.add_trace(
            go.Scatter(
                x=-pos["P"],
                y=pos["Mp"],
                mode="lines",
                name="SAP mirror",
                line=dict(color="#6b7280", width=1, dash="dot"),
            )
        )

    endpoints = pm_df[pm_df["Point"].isin(["Pmin", "Pmax"])]
    if not endpoints.empty:
        fig.add_trace(
            go.Scatter(
                x=endpoints["P"],
                y=[0.0] * len(endpoints),
                mode="markers",
                name="Axial limits",
                marker=dict(size=10, color="#dc2626", symbol="x"),
            )
        )

    fig.update_layout(
        title="P–M Interaction Curve",
        xaxis_title="Axial load P (kN)",
        yaxis_title="Moment capacity Mp (kNm)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_mc_single_with_hinge(
    mc_df: pd.DataFrame, hinge: HingeBackbone, P_target: float
) -> go.Figure:
    from src.analysis.pushover_derivation import select_P_curve

    curve = select_P_curve(mc_df, P_target)
    curve = curve[curve["curvature"] >= 0]
    backbone = pd.DataFrame(hinge.backbone_points)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["curvature"],
            y=curve["M"],
            mode="lines",
            name="M–φ curve",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backbone["curvature"],
            y=backbone["M"],
            mode="lines+markers",
            name="Bilinear backbone",
            line=dict(color="#dc2626", width=2, dash="dash"),
            marker=dict(size=8),
        )
    )
    fig.add_annotation(
        x=hinge.phi_y,
        y=hinge.My,
        text=f"My={hinge.My:.1f}",
        showarrow=True,
        arrowhead=2,
    )
    fig.add_annotation(
        x=hinge.phi_u,
        y=hinge.Mu,
        text=f"Mu={hinge.Mu:.1f}",
        showarrow=True,
        arrowhead=2,
    )
    fig.update_layout(
        title=f"Pushover Hinge Idealization (P ≈ {hinge.P:.0f} kN)",
        xaxis_title="Curvature (1/mm × 1000)",
        yaxis_title="Moment M (kNm)",
        height=500,
    )
    return fig
