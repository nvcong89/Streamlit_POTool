"""Unit tests for pushover hinge derivation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.pushover_derivation import derive_bilinear_hinge, select_P_curve


def _synthetic_bilinear_mc(
    P: float = -1000.0,
    My: float = 200.0,
    phi_y: float = 0.005,
    Mu: float = 350.0,
    phi_u: float = 0.015,
    K0: float = 50000.0,
    n_pre: int = 10,
    n_post: int = 10,
) -> pd.DataFrame:
    phi_pre = np.linspace(0.0005, phi_y, n_pre)
    M_pre = K0 * phi_pre
    phi_post = np.linspace(phi_y, phi_u, n_post + 1)[1:]
    Kp = (Mu - My) / (phi_u - phi_y)
    M_post = My + Kp * (phi_post - phi_y)
    phi = np.concatenate([phi_pre, phi_post])
    M = np.concatenate([M_pre, M_post])
    return pd.DataFrame(
        {
            "Point_Name": [f"Pt-{i}" for i in range(len(phi))],
            "P": P,
            "M": M,
            "e_cc": 0.0,
            "e_c": 0.0,
            "e_s": 0.0,
            "x": 500.0,
            "curvature": phi,
            "F_error": 0.0,
        }
    )


def test_select_P_curve_nearest():
    df = pd.concat([_synthetic_bilinear_mc(P=-1000), _synthetic_bilinear_mc(P=-500)])
    curve = select_P_curve(df, -950)
    assert curve["P"].iloc[0] == -1000


def test_derive_bilinear_hinge_yield_and_ultimate():
    df = _synthetic_bilinear_mc(My=200, phi_y=0.005, Mu=350, phi_u=0.015)
    hinge = derive_bilinear_hinge(df, P_target=-1000, stiffness_ratio=0.15)
    assert hinge.Mu == pytest.approx(350, rel=0.05)
    assert hinge.phi_u == pytest.approx(0.015, rel=0.05)
    assert hinge.My == pytest.approx(200, rel=0.15)
    assert hinge.K0 > 0
    assert len(hinge.backbone_points) == 3


def test_derive_bilinear_hinge_insufficient_points():
    df = _synthetic_bilinear_mc(n_pre=2, n_post=1)
    df = df[df["curvature"] > 0].head(3)
    with pytest.raises(ValueError, match="at least"):
        derive_bilinear_hinge(df, -1000)
