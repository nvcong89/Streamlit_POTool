"""Material models for moment-curvature analysis (Mander concrete, steel, bearing)."""

import numpy as np


class ConcreteModel:
    @staticmethod
    def unconfined_concrete_stress(
        e_c: float, e_c0: float, e_spall: float, r: float, f_ce: float
    ) -> float:
        e_cu0 = 2 * e_c0
        if e_c < 0:
            return 0.0
        if e_c <= e_cu0:
            ratio = e_c / e_c0
            return f_ce * ratio * (r / (r - 1 + ratio**r))
        if e_c <= e_spall:
            stress_at_ecu0 = (2 * f_ce * r) / (r - 1 + 2**r)
            return stress_at_ecu0 * ((e_spall - e_c) / (e_spall - e_cu0))
        return 0.0

    @staticmethod
    def confined_concrete_stress(e_c: float, e_cc: float, r_c: float, f_cc: float) -> float:
        if e_c < 0:
            return 0.0
        ratio = e_c / e_cc
        return f_cc * ratio * r_c / (r_c - 1 + ratio**r_c)

    @staticmethod
    def unconfined_concrete_stress_vectorized(
        e_c: np.ndarray, e_c0: float, e_spall: float, r: float, f_ce: float
    ) -> np.ndarray:
        e_cu0 = 2 * e_c0
        stress = np.zeros_like(e_c)
        mask1 = (e_c > 0) & (e_c <= e_cu0)
        ratio = e_c[mask1] / e_c0
        stress[mask1] = f_ce * ratio * (r / (r - 1 + ratio**r))
        mask2 = (e_c > e_cu0) & (e_c <= e_spall)
        stress_at_ecu0 = (2 * f_ce * r) / (r - 1 + 2**r)
        stress[mask2] = stress_at_ecu0 * ((e_spall - e_c[mask2]) / (e_spall - e_cu0))
        return stress

    @staticmethod
    def confined_concrete_stress_vectorized(
        e_c: np.ndarray, e_cc: float, r_c: float, f_cc: float
    ) -> np.ndarray:
        stress = np.zeros_like(e_c)
        mask = e_c > 0
        ratio = e_c[mask] / e_cc
        stress[mask] = f_cc * ratio * r_c / (r_c - 1 + ratio**r_c)
        return stress


class SteelModel:
    @staticmethod
    def rebar_stress_linear(
        e_s: float,
        e_ye: float,
        e_sh: float,
        e_smd: float,
        f_ye: float,
        f_ue: float,
        Es: float,
    ) -> float:
        e_abs = abs(e_s)
        if e_abs <= e_ye:
            stress = Es * e_abs
        elif e_abs <= e_sh:
            stress = f_ye
        elif e_abs <= e_smd:
            ratio = (e_abs - e_sh) / (e_smd - e_sh)
            stress = f_ye + (f_ue - f_ye) * ratio
        else:
            stress = f_ue
        return stress if e_s >= 0 else -stress

    @staticmethod
    def rebar_stress_vectorized(
        strain: np.ndarray,
        e_ye: float,
        e_sh: float,
        e_smd: float,
        f_ye: float,
        f_ue: float,
        Es: float,
    ) -> np.ndarray:
        abs_strain = np.abs(strain)
        stress = np.zeros_like(strain)
        mask1 = abs_strain <= e_ye
        stress[mask1] = Es * abs_strain[mask1]
        mask2 = (abs_strain > e_ye) & (abs_strain <= e_sh)
        stress[mask2] = f_ye
        mask3 = (abs_strain > e_sh) & (abs_strain <= e_smd)
        ratio = (abs_strain[mask3] - e_sh) / (e_smd - e_sh)
        stress[mask3] = f_ye + (f_ue - f_ye) * ratio
        mask4 = abs_strain > e_smd
        stress[mask4] = f_ue
        return stress * np.sign(strain)


class BearingPlateModel:
    @staticmethod
    def bearing_stress(e_s: float, e_ye: float, f_ye: float, Es: float) -> float:
        e_abs = abs(e_s)
        stress = Es * e_abs if e_abs <= e_ye else f_ye
        return stress if e_s >= 0 else -stress

    @staticmethod
    def bearing_stress_vectorized(
        strain: np.ndarray, e_ye: float, f_ye: float, Es: float
    ) -> np.ndarray:
        abs_strain = np.abs(strain)
        stress = np.where(abs_strain <= e_ye, Es * abs_strain, f_ye)
        return stress * np.sign(strain)
