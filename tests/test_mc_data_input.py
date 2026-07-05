"""Tests for MC_Data Excel input parser (A1:S20)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io.csv_schemas import parse_mc_data_input, parse_mc_data_xlsm

REFERENCE_XLSM = Path(
    r"c:\Users\210608\Documents\GitHub\py_Moment_Curvature_Analyser\outputs\pile_plug"
    r"\MC Analysis-Pile Plug-Ver 2.1 - AS BridgeManual SPM022_result.xlsm"
)
TEMPLATE_XLSM = Path(
    r"c:\Users\210608\Documents\GitHub\ExcelAddin-CN-Utilities\CN_VSTOProject\bin\Debug\MC"
    r"\MC Analysis-Pile Plug-Ver 2.1 - AS BridgeManual SPM022.xlsm"
)


@pytest.fixture(scope="module")
def spm022_section():
    if not REFERENCE_XLSM.exists():
        pytest.skip(f"Reference workbook not found: {REFERENCE_XLSM}")
    return parse_mc_data_input(REFERENCE_XLSM)


def test_parse_mc_data_input_spm022(spm022_section):
    s = spm022_section
    assert s["D"] == pytest.approx(680)
    assert s["f_ce"] == pytest.approx(40)
    assert s["f_ce"] != pytest.approx(52)
    assert s["f_ye"] == pytest.approx(550)
    assert s["e_ye_bearing"] == pytest.approx(0.001269125, rel=0.001)
    assert s["f_ye_bearing"] == pytest.approx(253.825, rel=0.001)
    assert s["Pmin"] == pytest.approx(-10616.07, rel=0.001)
    assert s["Pmax"] == pytest.approx(28365.63, rel=0.001)
    assert s["Lp"] == pytest.approx(1.083, rel=0.01)
    assert s["ec_MD"] == pytest.approx(0.004973, rel=0.001)
    assert s["ec_CD"] == pytest.approx(0.008574, rel=0.001)
    assert s["ec_ED"] == pytest.approx(0.012862, rel=0.001)
    assert s["n_bar"] == 12
    assert s["cover"] == pytest.approx(65)


def test_parse_mc_data_input_only_no_mc():
    if not REFERENCE_XLSM.exists():
        pytest.skip(f"Reference workbook not found: {REFERENCE_XLSM}")
    data = parse_mc_data_xlsm(REFERENCE_XLSM, inputs_only=True)
    assert data["section_input"]["D"] == pytest.approx(680)
    assert data["mc_results"] is None
    assert data["pm_summary"] is None


def test_parse_mc_data_input_template():
    if not TEMPLATE_XLSM.exists():
        pytest.skip(f"Template workbook not found: {TEMPLATE_XLSM}")
    section = parse_mc_data_input(TEMPLATE_XLSM)
    assert section["D"] > 0
    assert section["P_start"] < section["P_end"]
    assert section["e_ye_bearing"] > 0


def test_parse_mc_data_input_matches_xlsm_section(spm022_section):
    if not REFERENCE_XLSM.exists():
        pytest.skip(f"Reference workbook not found: {REFERENCE_XLSM}")
    full = parse_mc_data_xlsm(REFERENCE_XLSM)
    for key in ("D", "f_ce", "f_ye", "Pmin", "Pmax", "Lp", "e_ye_bearing"):
        assert spm022_section[key] == pytest.approx(full["section_input"][key])
