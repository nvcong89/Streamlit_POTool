"""Tests for P-M interaction curve derivation (Excel parity)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.mc_runner import create_excel_pm_P_array
from src.analysis.pm_curve import (
    PM_TABLE_COLUMNS,
    build_sap_pm3,
    derive_pm_table,
    strain_limits_from_section,
)
from src.io.csv_schemas import parse_mc_data_xlsm

REFERENCE_XLSM = Path(
    r"c:\Users\210608\Documents\GitHub\py_Moment_Curvature_Analyser\outputs\pile_plug"
    r"\MC Analysis-Pile Plug-Ver 2.1 - AS BridgeManual SPM022_result.xlsm"
)


@pytest.fixture(scope="module")
def reference_data():
    if not REFERENCE_XLSM.exists():
        pytest.skip(f"Reference workbook not found: {REFERENCE_XLSM}")
    return parse_mc_data_xlsm(REFERENCE_XLSM)


@pytest.fixture(scope="module")
def derived_pm(reference_data):
    section = reference_data["section_input"]
    mc = reference_data["mc_results"]
    ref = reference_data["pm_summary"]
    Lp = float(section["Lp"])
    limits = strain_limits_from_section(section)
    P_grid = ref["P"].tolist()
    return derive_pm_table(mc, limits, Lp, P_grid=P_grid)


def test_qp_formulas(derived_pm, reference_data):
    Lp = float(reference_data["section_input"]["Lp"])
    for _, row in derived_pm.iterrows():
        fy = row["Fy"]
        assert row["qp_mOLE"] == pytest.approx(Lp * max(0.0, row["FmOLE"] - fy), abs=1e-9)
        assert row["qp_mCLE"] == pytest.approx(Lp * max(0.0, row["FmCLE"] - fy), abs=1e-9)
        assert row["qp_mDE"] == pytest.approx(Lp * max(0.0, row["FmDE"] - fy), abs=1e-9)
        assert row["qUltimate"] == pytest.approx(Lp * (row["Fultimate"] - fy), abs=1e-9)


def test_pm_table_vs_reference(derived_pm, reference_data):
    ref = reference_data["pm_summary"]
    for label in ("P0", "P28", "P55"):
        r = ref[ref["Point"] == label].iloc[0]
        d = derived_pm[derived_pm["Point"] == label].iloc[0]
        assert d["Mp"] == pytest.approx(r["Mp"], rel=0.02)
        assert d["FmOLE"] == pytest.approx(r["FmOLE"], rel=0.02)
        assert d["FmCLE"] == pytest.approx(r["FmCLE"], rel=0.02)
        assert d["FmDE"] == pytest.approx(r["FmDE"], rel=0.02)
        # Fy follows Excel VBA logic; best-effort from M-φ strains (wider tolerance).
        assert d["Fy"] == pytest.approx(r["Fy"], rel=0.35)


def test_sap_pm3_mirror(derived_pm, reference_data):
    section = reference_data["section_input"]
    sap = build_sap_pm3(
        derived_pm,
        hinge_name="TestHinge",
        Pmin_extreme=float(section["Pmin"]),
        Pmax_extreme=float(section["Pmax"]),
    )
    assert len(sap) == 60
    assert sap.iloc[0]["M3"] == pytest.approx(0.0, abs=1e-9)
    assert sap.iloc[-1]["M3"] == pytest.approx(0.0, abs=1e-9)
    assert sap.iloc[0]["P"] == pytest.approx(-abs(section["Pmax"]), rel=0.001)
    assert sap.iloc[-1]["P"] == pytest.approx(-section["Pmin"], rel=0.001)

    pm_row = derived_pm.iloc[len(derived_pm) - 1 - 1]
    assert sap.iloc[1]["P"] == pytest.approx(-pm_row["P"], rel=0.001)
    assert sap.iloc[1]["M3"] == pytest.approx(pm_row["Mp"], rel=0.02)


def test_xlsm_import(reference_data):
    section = reference_data["section_input"]
    mc = reference_data["mc_results"]
    ref = reference_data["pm_summary"]
    assert section["D"] == pytest.approx(680, rel=0.01)
    assert len(mc) > 0
    assert len(ref) == 60
    assert list(ref.columns) == [
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


def test_excel_pm_p_grid_count():
    P = create_excel_pm_P_array(-10616, 28365, -3184.8, 14182.8, 12, 46)
    assert len(P) == 60
    assert P[0] == pytest.approx(-10616)
    assert P[-1] == pytest.approx(28365)


def test_pm_table_columns(derived_pm):
    assert list(derived_pm.columns) == PM_TABLE_COLUMNS
