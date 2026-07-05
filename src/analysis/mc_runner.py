"""Moment-curvature sweep for pile-plug sections."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.analysis.section_analysis import (
    MC_RESULT_COLUMNS,
    SectionAnalysis,
    compute_derived_geometry,
    validate_section_input,
)


def create_P_array(P_start: float, P_end: float, n_Pmin: int = 12, n_Pmax: int = 46) -> np.ndarray:
    delta_pmin = abs(P_start) / (n_Pmin - 1) if n_Pmin > 1 else 0
    delta_pmax = abs(P_end) / n_Pmax if n_Pmax > 0 else 0
    total_points = n_Pmin + n_Pmax
    arr = np.zeros(total_points)
    arr[0] = P_start
    for i in range(1, total_points):
        if i < n_Pmin:
            arr[i] = arr[i - 1] + delta_pmin
        else:
            arr[i] = arr[i - 1] + delta_pmax
    return arr


def create_strain_array(e_max: float, n_steps: int = 100) -> np.ndarray:
    return np.linspace(0.0, e_max, n_steps)


def run_mc_analysis(
    input_data: Dict,
    n_Pmin: int = 12,
    n_Pmax: int = 46,
    n_strain_steps: int = 100,
    progress_callback=None,
) -> List[Dict]:
    """
    Run full P–strain sweep and return list of M-φ result dicts.

    progress_callback: optional callable(completed: int, total: int)
    """
    data = compute_derived_geometry(input_data)
    errors = validate_section_input(data)
    if errors:
        raise ValueError("Invalid section input:\n" + "\n".join(f"  - {e}" for e in errors))

    section = SectionAnalysis()
    P_array = create_P_array(data["P_start"], data["P_end"], n_Pmin, n_Pmax)
    strain_array = create_strain_array(data["e_cmax_model"], n_strain_steps)

    results: List[Dict] = []
    total = len(P_array) * len(strain_array)
    completed = 0

    for i, P_i in enumerate(P_array):
        for j, e_c_j in enumerate(strain_array):
            point_id = f"P{i}-{j}"
            if j == 0:
                result = {
                    "Point_Name": point_id,
                    "P": float(P_i),
                    "M": 0.0,
                    "e_cc": 0.0,
                    "e_c": 0.0,
                    "e_s": 0.0,
                    "x": 0.0,
                    "curvature": 0.0,
                    "F_error": 0.0,
                }
            else:
                _, res = section.find_neutral_axis(float(P_i), float(e_c_j), data)
                result = {
                    "Point_Name": point_id,
                    "P": float(P_i),
                    "M": res["M"],
                    "e_cc": res["e_cc"],
                    "e_c": float(e_c_j),
                    "e_s": res["e_s_max"],
                    "x": res["x"],
                    "curvature": res["curvature"],
                    "F_error": res["F_error"],
                }
            results.append(result)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    return results


def results_to_dataframe(results: List[Dict]):
    import pandas as pd

    return pd.DataFrame(results, columns=MC_RESULT_COLUMNS)
