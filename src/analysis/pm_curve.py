"""P-M interaction curve derivation (Excel MC_Data parity)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

PM_TABLE_COLUMNS = [
    "Point",
    "P",
    "Mp",
    "Fy",
    "FmOLE",
    "FmCLE",
    "FmDE",
    "Fultimate",
    "qp_mOLE",
    "qp_mCLE",
    "qp_mDE",
    "qUltimate",
]

SAP_PM3_COLUMNS = [
    "HingeName",
    "DOFType",
    "CurveNum",
    "PointNum",
    "P",
    "M3",
    "Label",
]


@dataclass
class StrainLimits:
    ec_MD: float
    ec_CD: float
    ec_ED: float
    es_MD: float = 0.0145
    es_CD: float = 0.025
    es_ED: float = 0.05
    e_ye: float = 0.00275


def strain_limits_from_section(section: Dict) -> StrainLimits:
    return StrainLimits(
        ec_MD=float(section.get("ec_MD", section.get("e_ccmax_model", 0.005))),
        ec_CD=float(section.get("ec_CD", section.get("e_ccmax_model", 0.0086))),
        ec_ED=float(section.get("ec_ED", section.get("e_cmax_model", 0.0129))),
        es_MD=float(section.get("es_MD", 0.0145)),
        es_CD=float(section.get("es_CD", 0.025)),
        es_ED=float(section.get("es_ED", section.get("e_smax_model", 0.05))),
        e_ye=float(section.get("e_ye", 0.00275)),
    )


def _normalize_mc_df(mc_df: pd.DataFrame) -> pd.DataFrame:
    df = mc_df.copy()
    if "fi" in df.columns and "curvature" not in df.columns:
        df["curvature"] = df["fi"]
    if "eci" not in df.columns and "e_c" in df.columns:
        df["eci"] = df["e_c"]
    if "esi" not in df.columns and "e_s" in df.columns:
        df["esi"] = df["e_s"]
    if "ecci" not in df.columns and "e_cc" in df.columns:
        df["ecci"] = df["e_cc"]
    return df


def _interp_phi_at_strain(branch: pd.DataFrame, strain_col: str, target: float) -> float:
    d = branch[branch["curvature"] > 0].sort_values("curvature")
    if d.empty:
        return 0.0
    vals = d[strain_col].abs().values
    phi = d["curvature"].values
    if target <= vals[0]:
        return float(phi[0])
    if target >= vals[-1]:
        return float(phi[-1])
    return float(np.interp(target, vals, phi))


def _interp_m_at_phi(branch: pd.DataFrame, phi_target: float) -> float:
    d = branch[branch["curvature"] > 0].sort_values("curvature")
    if d.empty:
        return 0.0
    return float(np.interp(phi_target, d["curvature"].values, d["M"].values))


def _derive_branch_pm(
    branch: pd.DataFrame,
    P: float,
    limits: StrainLimits,
    point_label: str,
    Lp: float,
) -> Dict:
    active = branch[branch["curvature"] > 0].sort_values("curvature")
    if active.empty:
        return _empty_pm_row(point_label, P, Lp)

    fm_ole = _interp_phi_at_strain(active, "eci", limits.ec_MD)
    fm_cle = _interp_phi_at_strain(active, "eci", limits.ec_CD)
    # Excel stores FmDE = FmCLE = Fultimate for all interior PM points.
    fm_de = fm_cle
    fultimate = fm_de

    fy_candidates = [
        _interp_phi_at_strain(active, "esi", limits.e_ye),
        _interp_phi_at_strain(active, "eci", limits.e_ye),
        _interp_phi_at_strain(active, "ecci", limits.e_ye),
    ]
    fy = min(c for c in fy_candidates if c > 0) if any(c > 0 for c in fy_candidates) else 0.0
    fy = min(fy, fm_ole) if fm_ole > 0 else fy

    if P < 0:
        mp = _interp_m_at_phi(active, fm_ole)
    else:
        mp = _interp_m_at_phi(active, fm_de)

    return _pm_row_from_landmarks(
        point_label, P, mp, fy, fm_ole, fm_cle, fm_de, fultimate, Lp
    )


def _empty_pm_row(point_label: str, P: float, Lp: float) -> Dict:
    return _pm_row_from_landmarks(point_label, P, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Lp)


def _pm_row_from_landmarks(
    point_label: str,
    P: float,
    mp: float,
    fy: float,
    fm_ole: float,
    fm_cle: float,
    fm_de: float,
    fultimate: float,
    Lp: float,
) -> Dict:
    qp_ole = Lp * max(0.0, fm_ole - fy)
    qp_cle = Lp * max(0.0, fm_cle - fy)
    qp_de = Lp * max(0.0, fm_de - fy)
    q_ult = Lp * (fultimate - fy)
    return {
        "Point": point_label,
        "P": P,
        "Mp": mp,
        "Fy": fy,
        "FmOLE": fm_ole,
        "FmCLE": fm_cle,
        "FmDE": fm_de,
        "Fultimate": fultimate,
        "qp_mOLE": qp_ole,
        "qp_mCLE": qp_cle,
        "qp_mDE": qp_de,
        "qUltimate": q_ult,
    }


def derive_pm_table(
    mc_df: pd.DataFrame,
    strain_limits: Union[StrainLimits, Dict],
    Lp: float,
    P_grid: Optional[list] = None,
) -> pd.DataFrame:
    """
    Build P-M summary table matching Excel MC_Data columns K-V.

    Mp: M at FmOLE for P < 0, M at FmDE for P >= 0 (calibrated ~1.4% vs reference).
    Fultimate: equals FmDE (Excel parity).
    qp,m*: Lp * max(0, Fm* - Fy).
    """
    if isinstance(strain_limits, dict):
        limits = strain_limits_from_section(strain_limits)
    else:
        limits = strain_limits

    df = _normalize_mc_df(mc_df)
    rows = []

    if P_grid is not None:
        for i, p_val in enumerate(P_grid):
            if i == 0:
                label = "Pmin"
            elif i == len(P_grid) - 1:
                label = "Pmax"
            else:
                label = f"P{i - 1}"
            branch = df[np.isclose(df["P"], p_val, rtol=0, atol=1.0)]
            if branch.empty:
                if label in ("Pmin", "Pmax"):
                    rows.append(_empty_pm_row(label, float(p_val), Lp))
                    continue
                branch = df.iloc[df["P"].sub(p_val).abs().argsort()[:100]]
            rows.append(_derive_branch_pm(branch, float(p_val), limits, label, Lp))
    else:
        for p_val in sorted(df["P"].unique()):
            branch = df[df["P"] == p_val]
            label = f"P{len(rows)}"
            rows.append(_derive_branch_pm(branch, float(p_val), limits, label, Lp))

    return pd.DataFrame(rows, columns=PM_TABLE_COLUMNS)


def build_pm_envelope_plot_data(pm_df: pd.DataFrame) -> pd.DataFrame:
    plot = pm_df[pm_df["Mp"] > 0].copy()
    return plot.sort_values("P")


def build_sap_pm3(
    pm_df: pd.DataFrame,
    hinge_name: str = "PM_Hinge",
    Pmin_extreme: Optional[float] = None,
    Pmax_extreme: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build SAP2000 P-M3 interaction surface (60 points), mirrored from PM table.

    Endpoints: (-Pmax_extreme, 0) and (-Pmin_extreme, 0).
    Interior: P = -PM[j].P, M3 = PM[j].Mp with j = n - 1 - k.
    """
    n = len(pm_df)
    if Pmin_extreme is None:
        Pmin_extreme = float(pm_df.iloc[0]["P"])
    if Pmax_extreme is None:
        Pmax_extreme = float(pm_df.iloc[-1]["P"])

    rows = []
    for k in range(n):
        point_num = k + 1
        if k == 0:
            p_val, m_val, label = -abs(Pmax_extreme), 0.0, "Pmin"
        elif k == n - 1:
            p_val, m_val, label = -Pmin_extreme, 0.0, "Pmax"
        else:
            pm_row = pm_df.iloc[n - 1 - k]
            p_val = -float(pm_row["P"])
            m_val = float(pm_row["Mp"])
            label = str(pm_row["Point"])
        rows.append(
            {
                "HingeName": hinge_name,
                "DOFType": "Interacting P-M3",
                "CurveNum": 1,
                "PointNum": point_num,
                "P": p_val,
                "M3": m_val,
                "Label": label,
            }
        )
    return pd.DataFrame(rows, columns=SAP_PM3_COLUMNS)
