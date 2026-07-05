"""Derive bilinear pushover hinge backbone from M-φ curves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class HingeBackbone:
    P: float
    My: float
    phi_y: float
    Mu: float
    phi_u: float
    K0: float
    Kp: float
    backbone_points: List[Dict[str, float]] = field(default_factory=list)


def select_P_curve(mc_df: pd.DataFrame, P_target: float, tol: float = 1.0) -> pd.DataFrame:
    """Return M-φ points for the axial load level nearest to P_target."""
    if mc_df.empty:
        raise ValueError("M-φ dataframe is empty")
    unique_P = mc_df["P"].unique()
    nearest_P = unique_P[np.argmin(np.abs(unique_P - P_target))]
    curve = mc_df[np.isclose(mc_df["P"], nearest_P, atol=tol)].copy()
    if curve.empty:
        curve = mc_df[mc_df["P"] == nearest_P].copy()
    return curve.sort_values("curvature").reset_index(drop=True)


def derive_bilinear_hinge(
    mc_df: pd.DataFrame,
    P_target: float,
    stiffness_ratio: float = 0.15,
    min_points: int = 5,
) -> HingeBackbone:
    """
    Idealize M-φ curve at axial load P into a bilinear hinge backbone.

    stiffness_ratio: yield when secant stiffness drops below this fraction of K0.
    """
    curve = select_P_curve(mc_df, P_target)
    curve = curve[curve["curvature"] > 0].copy()
    if len(curve) < min_points:
        raise ValueError(f"Need at least {min_points} points with curvature > 0 for P ≈ {P_target}")

    phi = curve["curvature"].to_numpy()
    M = curve["M"].to_numpy()

    Mu_idx = int(np.argmax(M))
    Mu = float(M[Mu_idx])
    phi_u = float(phi[Mu_idx])

    # Initial stiffness from pre-peak elastic segment (M < 75% of Mu)
    elastic_mask = M <= 0.75 * Mu
    elastic_indices = np.where(elastic_mask)[0]
    n_init = max(3, min(len(elastic_indices), 8))
    if len(elastic_indices) >= 2:
        idx = elastic_indices[:n_init]
        K0 = float(np.polyfit(phi[idx], M[idx], 1)[0])
    else:
        n_fit = max(3, min(5, len(phi) // 4))
        K0 = float(np.polyfit(phi[:n_fit], M[:n_fit], 1)[0])
    if K0 <= 0:
        K0 = float(M[1] / phi[1]) if phi[1] > 0 else 1.0

    threshold = stiffness_ratio * K0
    My = float(M[min(1, len(M) - 1)])
    phi_y = float(phi[min(1, len(phi) - 1)])

    for i in range(2, Mu_idx + 1):
        if phi[i] - phi[i - 1] <= 0:
            continue
        tangent = (M[i] - M[i - 1]) / (phi[i] - phi[i - 1])
        if tangent < threshold:
            My = float(M[i])
            phi_y = float(phi[i])
            break
    else:
        # Fallback: yield at 75% of ultimate on initial stiffness line
        My = 0.75 * Mu
        phi_y = My / K0 if K0 > 0 else float(phi[1])

    if phi_u > phi_y:
        Kp = (Mu - My) / (phi_u - phi_y)
    else:
        Kp = 0.0

    P_used = float(curve["P"].iloc[0])
    backbone_points = [
        {"step": 0, "M": 0.0, "curvature": 0.0},
        {"step": 1, "M": My, "curvature": phi_y},
        {"step": 2, "M": Mu, "curvature": phi_u},
    ]

    return HingeBackbone(
        P=P_used,
        My=My,
        phi_y=phi_y,
        Mu=Mu,
        phi_u=phi_u,
        K0=K0,
        Kp=Kp,
        backbone_points=backbone_points,
    )


def hinge_to_summary_df(hinge: HingeBackbone) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "P": hinge.P,
                "My": hinge.My,
                "phi_y": hinge.phi_y,
                "Mu": hinge.Mu,
                "phi_u": hinge.phi_u,
                "K0": hinge.K0,
                "Kp": hinge.Kp,
            }
        ]
    )


def hinge_to_backbone_df(hinge: HingeBackbone) -> pd.DataFrame:
    return pd.DataFrame(hinge.backbone_points)


def sample_material_curves(input_data: Dict, n_points: int = 200) -> pd.DataFrame:
    """Sample stress-strain curves from section material parameters."""
    from src.materials.material_models import BearingPlateModel, ConcreteModel, SteelModel

    rows = []
    e_max = max(
        input_data.get("e_cmax_model", 0.025),
        input_data.get("e_smax_model", 0.12),
        input_data.get("e_smd_bearing", 0.05),
    )
    strains = np.linspace(0, e_max, n_points)

    for e in strains:
        rows.append(
            {
                "material": "unconfined_concrete",
                "strain": e,
                "stress": ConcreteModel.unconfined_concrete_stress(
                    e,
                    input_data["e_c0"],
                    input_data["e_spall"],
                    input_data["r_c"],
                    input_data["f_ce"],
                ),
            }
        )
        rows.append(
            {
                "material": "confined_concrete",
                "strain": e,
                "stress": ConcreteModel.confined_concrete_stress(
                    e, input_data["e_cc"], input_data["r_cc"], input_data["f_cc"]
                ),
            }
        )
        rows.append(
            {
                "material": "rebar_steel",
                "strain": e,
                "stress": SteelModel.rebar_stress_linear(
                    e,
                    input_data["e_ye"],
                    input_data["e_sh"],
                    input_data["e_smd"],
                    input_data["f_ye"],
                    input_data["f_ue"],
                    input_data["Es"],
                ),
            }
        )
        rows.append(
            {
                "material": "bearing_steel",
                "strain": e,
                "stress": BearingPlateModel.bearing_stress(
                    e,
                    input_data["e_ye_bearing"],
                    input_data["f_ye_bearing"],
                    input_data["Es_bearing"],
                ),
            }
        )

    return pd.DataFrame(rows)
