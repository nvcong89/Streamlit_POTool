"""Pile-plug fiber section analysis (no Streamlit dependencies)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import bisect

from src.materials.material_models import BearingPlateModel, ConcreteModel, SteelModel

MC_RESULT_COLUMNS = [
    "Point_Name",
    "P",
    "M",
    "e_cc",
    "e_c",
    "e_s",
    "x",
    "curvature",
    "F_error",
]


def compute_derived_geometry(input_data: Dict) -> Dict:
    """Compute D0 (confined core) from section geometry."""
    D = input_data["D"]
    cover = input_data["cover"]
    phi_link = input_data["phi_link"]
    phi_main = input_data["phi_main"]
    data = dict(input_data)
    data["D0"] = D - 2 * cover - phi_link
    data["Ds"] = D - 2 * cover - 2 * phi_link - phi_main
    return data


def validate_section_input(input_data: Dict) -> List[str]:
    errors: List[str] = []
    D = input_data.get("D", 0)
    cover = input_data.get("cover", 0)
    phi_main = input_data.get("phi_main", 0)
    phi_link = input_data.get("phi_link", 0)
    n_bar = input_data.get("n_bar", 0)

    if D <= 0:
        errors.append("D (pile plug diameter) must be > 0")
    if input_data.get("D_bearing", 0) <= 0:
        errors.append("D_bearing must be > 0")
    if cover <= 0:
        errors.append("cover must be > 0")
    if phi_main <= 0:
        errors.append("phi_main must be > 0")
    if phi_link <= 0:
        errors.append("phi_link must be > 0")
    if n_bar < 3:
        errors.append("n_bar must be >= 3")
    if input_data.get("P_start", 0) > input_data.get("P_end", 0):
        errors.append("P_start must be <= P_end")
    if input_data.get("e_cmax_model", 0) <= 0:
        errors.append("e_cmax_model must be > 0")

    R0 = D / 2
    available = R0 - cover - phi_link - phi_main / 2
    if D > 0 and available <= 0:
        errors.append("Rebars do not fit in section (cover + link + rebar/2 > R0)")

    return errors


class SectionAnalysis:
    def __init__(self) -> None:
        self.concrete_model = ConcreteModel()
        self.steel_model = SteelModel()
        self.bearing_model = BearingPlateModel()

    def calculate_reinforcement_force_moment(
        self,
        x: float,
        e_c_max: float,
        phi_main: float,
        n_bar: int,
        phi_link: float,
        D: float,
        cover: float,
        e_ye: float,
        e_sh: float,
        e_smd: float,
        f_ye: float,
        f_ue: float,
        Es: float,
    ) -> Tuple[float, float]:
        A_bar = np.pi * (phi_main / 2) ** 2
        R = D / 2 - cover - phi_link - phi_main / 2
        R0 = D / 2
        theta_array = np.arange(n_bar) * (2 * np.pi / n_bar)
        d_bar_array = R0 - R * np.cos(theta_array)
        if x > 0:
            strain_array = e_c_max * (x - d_bar_array) / x
        else:
            strain_array = np.zeros_like(d_bar_array)
        stress_array = self.steel_model.rebar_stress_vectorized(
            strain_array, e_ye, e_sh, e_smd, f_ye, f_ue, Es
        )
        force_array = stress_array * A_bar
        moment_array = force_array * (R0 - d_bar_array)
        return np.sum(force_array) / 1000, np.sum(moment_array) / 1_000_000

    def calculate_concrete_force_moment(
        self,
        x: float,
        e_c_max: float,
        D: float,
        D0: float,
        e_c0: float,
        e_spall: float,
        r_c: float,
        f_ce: float,
        e_cc: float,
        r_cc: float,
        f_cc: float,
        n_layers: int = 100,
    ) -> Tuple[float, float, float, float]:
        dy = D / n_layers
        R0 = D / 2
        R_confined = D0 / 2
        y_array = np.linspace(dy / 2, D - dy / 2, n_layers)
        if x > 0:
            strain_array = e_c_max * (x - y_array) / x
        else:
            strain_array = np.zeros_like(y_array)
        mask_compression = strain_array > 0
        if not np.any(mask_compression):
            return 0.0, 0.0, 0.0, 0.0

        y_comp = y_array[mask_compression]
        strain_comp = strain_array[mask_compression]
        moment_arm = R0 - y_comp
        y_from_center = y_comp - R0
        b_total = 2 * np.sqrt(np.maximum(0, R0**2 - y_from_center**2))
        mask_confined = np.abs(y_from_center) <= R_confined
        b_confined = np.where(
            mask_confined,
            2 * np.sqrt(np.maximum(0, R_confined**2 - y_from_center**2)),
            0,
        )
        b_unconfined = b_total - b_confined
        stress_unconfined = self.concrete_model.unconfined_concrete_stress_vectorized(
            strain_comp, e_c0, e_spall, r_c, f_ce
        )
        stress_confined = self.concrete_model.confined_concrete_stress_vectorized(
            strain_comp, e_cc, r_cc, f_cc
        )
        dA_unconfined = b_unconfined * dy
        dA_confined = b_confined * dy
        dF_unconfined = stress_unconfined * dA_unconfined
        dF_confined = stress_confined * dA_confined
        dM_unconfined = dF_unconfined * moment_arm
        dM_confined = dF_confined * moment_arm
        return (
            np.sum(dF_unconfined) / 1000,
            np.sum(dM_unconfined) / 1_000_000,
            np.sum(dF_confined) / 1000,
            np.sum(dM_confined) / 1_000_000,
        )

    def calculate_steel_pipe_force_moment(
        self,
        x: float,
        e_c_max: float,
        D_bearing: float,
        D: float,
        e_ye: float,
        f_ye: float,
        Es: float,
        n_segments: int = 100,
    ) -> Tuple[float, float]:
        R_outer = D_bearing / 2
        R_inner = D / 2
        R0_bearing = D_bearing / 2
        t = R_outer - R_inner
        R_mid = (R_outer + R_inner) / 2
        theta_array = (np.arange(n_segments) + 0.5) * (2 * np.pi / n_segments)
        y_center_array = R_mid * np.cos(theta_array)
        y_array = R0_bearing + y_center_array
        if x > 0:
            strain_array = e_c_max * (x - y_array) / x
        else:
            strain_array = np.zeros_like(y_array)
        stress_array = self.bearing_model.bearing_stress_vectorized(strain_array, e_ye, f_ye, Es)
        delta_theta = 2 * np.pi / n_segments
        dA = R_mid * delta_theta * t
        dF_array = stress_array * dA
        moment_arm_array = R0_bearing - y_array
        dM_array = dF_array * moment_arm_array
        return np.sum(dF_array) / 1000, np.sum(dM_array) / 1_000_000

    def find_neutral_axis(
        self, P_target: float, e_c_max: float, input_data: Dict, tol: float = 1e-6
    ) -> Tuple[float, Dict]:
        D = input_data["D"]
        last_results: Dict = {}

        def force_balance(x_trial: float) -> float:
            F_steel, M_steel = self.calculate_reinforcement_force_moment(
                x_trial,
                e_c_max,
                input_data["phi_main"],
                int(input_data["n_bar"]),
                input_data["phi_link"],
                input_data["D"],
                input_data["cover"],
                input_data["e_ye"],
                input_data["e_sh"],
                input_data["e_smd"],
                input_data["f_ye"],
                input_data["f_ue"],
                input_data["Es"],
            )
            F_c_unconf, M_c_unconf, F_c_conf, M_c_conf = self.calculate_concrete_force_moment(
                x_trial,
                e_c_max,
                input_data["D"],
                input_data["D0"],
                input_data["e_c0"],
                input_data["e_spall"],
                input_data["r_c"],
                input_data["f_ce"],
                input_data["e_cc"],
                input_data["r_cc"],
                input_data["f_cc"],
            )
            F_bearing, M_bearing = self.calculate_steel_pipe_force_moment(
                x_trial,
                e_c_max,
                input_data["D_bearing"],
                input_data["D"],
                input_data["e_ye_bearing"],
                input_data["f_ye_bearing"],
                input_data["Es_bearing"],
            )
            last_results.update(
                {
                    "F_steel": F_steel,
                    "M_steel": M_steel,
                    "F_c_unconf": F_c_unconf,
                    "M_c_unconf": M_c_unconf,
                    "F_c_conf": F_c_conf,
                    "M_c_conf": M_c_conf,
                    "F_bearing": F_bearing,
                    "M_bearing": M_bearing,
                }
            )
            F_total = F_steel + F_c_unconf + F_c_conf + F_bearing
            return F_total - P_target

        try:
            x_solution = bisect(force_balance, 0.05 * D, 100 * D, xtol=tol, maxiter=1000)
        except ValueError:
            x_solution = D / 2

        M_total = (
            last_results["M_steel"]
            + last_results["M_c_unconf"]
            + last_results["M_c_conf"]
            + last_results["M_bearing"]
        )
        F_error = (
            last_results["F_steel"]
            + last_results["F_c_unconf"]
            + last_results["F_c_conf"]
            + last_results["F_bearing"]
            - P_target
        )

        d_max = input_data["D"] - input_data["cover"] - input_data["phi_link"] - input_data["phi_main"] / 2
        e_s_max = e_c_max * (x_solution - d_max) / x_solution if x_solution > 0 else 0.0
        d_confined = input_data["cover"] + input_data["phi_link"]
        e_cc_i = e_c_max * (x_solution - d_confined) / x_solution if x_solution > 0 else 0.0

        results = {
            "x": x_solution,
            "M": M_total,
            "e_s_max": e_s_max,
            "e_cc": e_cc_i,
            "curvature": e_c_max * 1000 / x_solution if x_solution > 0 else 0.0,
            "F_error": F_error,
        }
        return x_solution, results
