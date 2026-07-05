"""CSV schemas, loaders, and column mapping."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.analysis.section_analysis import MC_RESULT_COLUMNS

SECTION_INPUT_FIELDS = [
    "D",
    "D_bearing",
    "phi_main",
    "n_bar",
    "phi_link",
    "cover",
    "f_ce",
    "e_c0",
    "e_cu0",
    "e_spall",
    "r_c",
    "f_cu",
    "e_cc",
    "e_ccu",
    "r_cc",
    "f_cc",
    "e_ye",
    "e_sh",
    "e_smd",
    "f_ye",
    "f_ue",
    "Es",
    "e_ye_bearing",
    "e_sh_bearing",
    "e_smd_bearing",
    "f_ye_bearing",
    "f_ue_bearing",
    "Es_bearing",
    "P_start",
    "P_end",
    "e_ccmax_model",
    "e_cmax_model",
    "e_smax_model",
    "code",
]

MC_REQUIRED_FIELDS = ["P", "M", "curvature"]
MC_OPTIONAL_FIELDS = [c for c in MC_RESULT_COLUMNS if c not in MC_REQUIRED_FIELDS]

MATERIAL_CURVE_FIELDS = ["material", "strain", "stress"]

DEFAULT_SECTION_VALUES: Dict[str, float | str] = {
    "D": 1200.0,
    "D_bearing": 1220.0,
    "phi_main": 32.0,
    "n_bar": 20.0,
    "phi_link": 10.0,
    "cover": 75.0,
    "f_ce": 40.0,
    "e_c0": 0.002,
    "e_cu0": 0.004,
    "e_spall": 0.006,
    "r_c": 4.0,
    "f_cu": 0.0,
    "e_cc": 0.0035,
    "e_ccu": 0.015,
    "r_cc": 1.5,
    "f_cc": 48.0,
    "e_ye": 0.002,
    "e_sh": 0.005,
    "e_smd": 0.08,
    "f_ye": 500.0,
    "f_ue": 650.0,
    "Es": 200000.0,
    "e_ye_bearing": 0.002,
    "e_sh_bearing": 0.005,
    "e_smd_bearing": 0.05,
    "f_ye_bearing": 355.0,
    "f_ue_bearing": 510.0,
    "Es_bearing": 200000.0,
    "P_start": -8000.0,
    "P_end": 8000.0,
    "e_ccmax_model": 0.015,
    "e_cmax_model": 0.025,
    "e_smax_model": 0.12,
    "code": "ASCE 61-14",
}


def load_csv_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def parse_section_input(df: pd.DataFrame, column_map: Optional[Dict[str, str]] = None) -> Dict:
    """Parse section input from key-value or wide single-row CSV."""
    if column_map:
        df = df.rename(columns=column_map)

    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Key-value format: parameter, value
    if "parameter" in cols_lower and "value" in cols_lower:
        pcol = cols_lower["parameter"]
        vcol = cols_lower["value"]
        data = {}
        for _, row in df.iterrows():
            key = str(row[pcol]).strip()
            if key in SECTION_INPUT_FIELDS or key in DEFAULT_SECTION_VALUES:
                val = row[vcol]
                if key == "code":
                    data[key] = str(val)
                else:
                    data[key] = float(val)
        for k, v in DEFAULT_SECTION_VALUES.items():
            data.setdefault(k, v)
        return _coerce_section_types(data)

    # Wide single-row
    if len(df) == 1:
        row = df.iloc[0]
        data = {}
        for field in SECTION_INPUT_FIELDS:
            if field in df.columns:
                val = row[field]
                data[field] = str(val) if field == "code" else float(val)
            elif field.lower() in cols_lower:
                orig = cols_lower[field.lower()]
                val = row[orig]
                data[field] = str(val) if field == "code" else float(val)
        for k, v in DEFAULT_SECTION_VALUES.items():
            data.setdefault(k, v)
        return _coerce_section_types(data)

    raise ValueError(
        "Section input must be key-value (parameter, value) or a single wide row. "
        f"Got {len(df)} rows with columns: {list(df.columns)}"
    )


def _coerce_section_types(data: Dict) -> Dict:
    out = dict(data)
    out["n_bar"] = int(float(out.get("n_bar", 20)))
    for key in SECTION_INPUT_FIELDS:
        if key == "code":
            out.setdefault(key, "ASCE 61-14")
        elif key != "code" and key in out:
            out[key] = float(out[key])
        elif key != "code":
            out[key] = float(DEFAULT_SECTION_VALUES[key])  # type: ignore[arg-type]
    return out


def parse_mc_results(df: pd.DataFrame, column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    if column_map:
        df = df.rename(columns=column_map)
    missing = [f for f in MC_REQUIRED_FIELDS if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required M-φ columns: {missing}")
    out = df.copy()
    for col in MC_RESULT_COLUMNS:
        if col not in out.columns:
            if col == "Point_Name":
                out[col] = [f"Pt-{i}" for i in range(len(out))]
            elif col in ("e_cc", "e_c", "e_s", "x", "F_error"):
                out[col] = 0.0
            else:
                continue
    return out[MC_RESULT_COLUMNS]


def parse_material_curves(df: pd.DataFrame, column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    if column_map:
        df = df.rename(columns=column_map)
    missing = [f for f in MATERIAL_CURVE_FIELDS if f not in df.columns]
    if missing:
        raise ValueError(f"Missing material curve columns: {missing}")
    return df[MATERIAL_CURVE_FIELDS].copy()


def build_column_mapper(
    uploaded_columns: List[str], expected_fields: List[str]
) -> Tuple[Dict[str, str], List[str]]:
    """Auto-map columns by case-insensitive name match; return map and unmapped required."""
    mapping: Dict[str, str] = {}
    lower_to_orig = {c.lower().strip(): c for c in uploaded_columns}
    for field in expected_fields:
        if field in uploaded_columns:
            mapping[field] = field
        elif field.lower() in lower_to_orig:
            mapping[lower_to_orig[field.lower()]] = field
    unmapped = [f for f in expected_fields if f not in mapping.values()]
    return mapping, unmapped


def template_section_kv_csv() -> str:
    lines = ["parameter,value"]
    for k, v in DEFAULT_SECTION_VALUES.items():
        lines.append(f"{k},{v}")
    return "\n".join(lines)


def template_mc_results_csv() -> str:
    return ",".join(MC_RESULT_COLUMNS) + "\nP0-1,-1000,100,0.001,0.002,0.001,600,0.003,0.5\n"


def read_template(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "templates" / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    if name == "section_input.csv":
        return template_section_kv_csv()
    if name == "mc_results.csv":
        return template_mc_results_csv()
    raise FileNotFoundError(name)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
