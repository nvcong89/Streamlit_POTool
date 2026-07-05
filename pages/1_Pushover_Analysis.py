"""Pushover Analysis — main Streamlit tool page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.mc_runner import create_excel_pm_P_array, run_mc_analysis, results_to_dataframe
from src.analysis.pm_curve import build_sap_pm3, derive_pm_table, strain_limits_from_section
from src.analysis.pushover_derivation import (
    derive_bilinear_hinge,
    hinge_to_backbone_df,
    hinge_to_summary_df,
    sample_material_curves,
)
from src.analysis.section_analysis import compute_derived_geometry, validate_section_input
from src.io.csv_schemas import (
    DEFAULT_SECTION_VALUES,
    MC_REQUIRED_FIELDS,
    SECTION_INPUT_FIELDS,
    build_column_mapper,
    df_to_csv_bytes,
    load_csv_upload,
    parse_material_curves,
    parse_mc_results,
    parse_mc_data_xlsm,
    parse_mc_data_input,
    parse_section_input,
    template_mc_results_csv,
    template_section_kv_csv,
)
from src.ui.charts import (
    plot_mc_family,
    plot_mc_single_with_hinge,
    plot_pm_curve,
    plot_stress_strain,
)

st.set_page_config(page_title="Pushover Analysis", layout="wide")
st.title("Pushover Analysis")

# Session state defaults
if "section_input" not in st.session_state:
    st.session_state.section_input = dict(DEFAULT_SECTION_VALUES)
if "mc_results" not in st.session_state:
    st.session_state.mc_results = None
if "material_curves" not in st.session_state:
    st.session_state.material_curves = None
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Section input (run M-φ)"
if "hinge" not in st.session_state:
    st.session_state.hinge = None
if "pm_results" not in st.session_state:
    st.session_state.pm_results = None
if "sap_pm3" not in st.session_state:
    st.session_state.sap_pm3 = None
if "pm_reference" not in st.session_state:
    st.session_state.pm_reference = None

with st.sidebar:
    st.header("Settings")
    input_mode = st.radio(
        "Input mode",
        [
            "Section input (run M-φ)",
            "Pre-computed M-φ",
            "Material curves only",
        ],
        index=[
            "Section input (run M-φ)",
            "Pre-computed M-φ",
            "Material curves only",
        ].index(st.session_state.input_mode),
    )
    st.session_state.input_mode = input_mode

    code = st.selectbox("Code / standard", ["ASCE 61-14", "Turkish Code 2020"])
    if st.session_state.section_input:
        st.session_state.section_input["code"] = code

    st.subheader("M-φ sweep (mode A)")
    n_pmin = st.number_input("P min steps", min_value=4, max_value=30, value=12)
    n_pmax = st.number_input("P max steps", min_value=10, max_value=80, value=20)
    n_strain = st.number_input("Strain steps", min_value=20, max_value=200, value=50)
    use_excel_pm_grid = st.checkbox(
        "Use Excel 60-point P grid (Pmin…Pmax)",
        value=True,
        help="Matches MC_Data PM table: Pmin + interior sweep + Pmax.",
    )

    st.subheader("Hinge idealization")
    stiffness_ratio = st.slider(
        "Yield stiffness ratio (fraction of K0)",
        min_value=0.05,
        max_value=0.40,
        value=0.15,
        step=0.01,
    )

tab_upload, tab_material, tab_mc, tab_pm, tab_pushover, tab_export = st.tabs(
    ["Upload & Map", "Material", "Moment–Curvature", "P–M Curve", "Pushover Derivation", "Export"]
)


def _apply_section_input(section: dict, *, clear_mc: bool = True) -> None:
    section["code"] = code
    st.session_state.section_input = section
    st.session_state.material_curves = sample_material_curves(section)
    st.session_state.hinge = None
    st.session_state.pm_results = None
    st.session_state.sap_pm3 = None
    st.session_state.pm_reference = None
    if clear_mc:
        st.session_state.mc_results = None


def _section_input_preview_df(section: dict) -> pd.DataFrame:
    rows = [{"parameter": f, "value": section.get(f, "")} for f in SECTION_INPUT_FIELDS]
    return pd.DataFrame(rows)


with tab_upload:
    st.subheader("Upload data")
    uploaded = st.file_uploader(
        "CSV, Excel, or MC workbook (.xlsm)",
        type=["csv", "xlsx", "xls", "xlsm"],
        help="Use templates from the home page, MC_Data .xlsm export, or your own column names.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Section input template",
            template_section_kv_csv(),
            file_name="section_input.csv",
            mime="text/csv",
        )
    with col_b:
        st.download_button(
            "M-φ results template",
            template_mc_results_csv(),
            file_name="mc_results.csv",
            mime="text/csv",
        )

    if uploaded:
        try:
            name_lower = uploaded.name.lower()
            is_mc_workbook = name_lower.endswith((".xlsm", ".xlsx"))

            if is_mc_workbook:
                inputs_only = st.checkbox(
                    "Chỉ đọc Input (A1:S20)",
                    value=False,
                    help="Load section parameters only; skip M-φ and PM results from the workbook.",
                )
                xlsm_data = parse_mc_data_xlsm(uploaded, inputs_only=inputs_only)

                if inputs_only:
                    st.success("MC_Data input region loaded (A1:S20).")
                    with st.expander("Preview section input from Excel"):
                        st.dataframe(
                            _section_input_preview_df(xlsm_data["section_input"]),
                            use_container_width=True,
                        )
                    if input_mode == "Section input (run M-φ)":
                        if st.button("Apply section input", type="primary", key="apply_mc_inputs_only"):
                            _apply_section_input(xlsm_data["section_input"], clear_mc=True)
                            st.success("Section input loaded from MC_Data. Run M-φ on Moment–Curvature tab.")
                    else:
                        st.info("Switch to 'Section input (run M-φ)' mode to apply Excel section input.")
                else:
                    mc_n = len(xlsm_data["mc_results"]) if xlsm_data["mc_results"] is not None else 0
                    pm_n = (
                        len(xlsm_data["pm_summary"])
                        if xlsm_data["pm_summary"] is not None
                        else 0
                    )
                    st.success(f"Loaded MC workbook: {mc_n} M-φ rows, {pm_n} PM reference rows.")
                    with st.expander("Preview section input from Excel"):
                        st.dataframe(
                            _section_input_preview_df(xlsm_data["section_input"]),
                            use_container_width=True,
                        )
                    if st.button("Apply MC workbook", type="primary"):
                        section = dict(xlsm_data["section_input"])
                        section["code"] = code
                        st.session_state.section_input = section
                        st.session_state.mc_results = xlsm_data["mc_results"]
                        st.session_state.pm_reference = xlsm_data["pm_summary"]
                        st.session_state.material_curves = sample_material_curves(section)
                        st.session_state.pm_results = None
                        st.session_state.sap_pm3 = None
                        st.session_state.hinge = None
                        st.success("MC workbook loaded (section, M-φ, optional PM reference).")
                    if xlsm_data["mc_results"] is not None:
                        with st.expander("Preview M-φ data"):
                            st.dataframe(xlsm_data["mc_results"].head(20), use_container_width=True)
                    if xlsm_data["pm_summary"] is not None:
                        with st.expander("Preview Excel PM summary (compare)"):
                            st.dataframe(xlsm_data["pm_summary"].head(20), use_container_width=True)
            else:
                raw_df = load_csv_upload(uploaded)
                st.success(f"Loaded {len(raw_df)} rows, columns: {list(raw_df.columns)}")

                if input_mode == "Section input (run M-φ)":
                    expected = SECTION_INPUT_FIELDS
                elif input_mode == "Pre-computed M-φ":
                    expected = MC_REQUIRED_FIELDS
                else:
                    expected = ["material", "strain", "stress"]

                auto_map, unmapped = build_column_mapper(list(raw_df.columns), expected)
                st.subheader("Column mapper")
                st.caption("Adjust mapping if your headers differ from expected names.")
                user_map = {}
                for col in raw_df.columns:
                    default_target = auto_map.get(col, "(ignore)")
                    options = ["(ignore)"] + expected
                    idx = options.index(default_target) if default_target in options else 0
                    target = st.selectbox(f"`{col}` →", options, index=idx, key=f"map_{col}")
                    if target != "(ignore)":
                        user_map[col] = target

                inv_map = {v: k for k, v in user_map.items()}
                rename_for_parse = {inv_map[f]: f for f in expected if f in inv_map}

                if st.button("Apply upload", type="primary"):
                    if input_mode == "Section input (run M-φ)":
                        section = parse_section_input(raw_df, rename_for_parse or None)
                        section["code"] = code
                        st.session_state.section_input = section
                        st.session_state.material_curves = sample_material_curves(section)
                        st.success("Section input loaded. Go to Moment–Curvature to run analysis.")
                    elif input_mode == "Pre-computed M-φ":
                        mc_df = parse_mc_results(raw_df, rename_for_parse or None)
                        st.session_state.mc_results = mc_df
                        st.session_state.pm_results = None
                        st.success(f"Loaded {len(mc_df)} M-φ points.")
                    else:
                        mat_df = parse_material_curves(raw_df, rename_for_parse or None)
                        st.session_state.material_curves = mat_df
                        st.success(f"Loaded {len(mat_df)} material curve points.")

                with st.expander("Preview raw data"):
                    st.dataframe(raw_df.head(20), use_container_width=True)

        except Exception as exc:
            st.error(str(exc))

    with st.expander("Optional: MC Excel input (MC_Data A1:S20)"):
        st.caption(
            "Upload a template or result workbook to load section parameters only "
            "(geometry, materials, strain limits, P grid)."
        )
        mc_input_file = st.file_uploader(
            "MC_Data input Excel",
            type=["xlsm", "xlsx"],
            key="mc_input_only",
            help="Reads MC_Data sheet rows 4–19 (A1:S20 input region).",
        )
        if mc_input_file:
            try:
                parsed_section = parse_mc_data_input(mc_input_file)
                st.dataframe(_section_input_preview_df(parsed_section), use_container_width=True)
                if input_mode == "Section input (run M-φ)":
                    if st.button("Load section from Excel", type="primary", key="load_mc_input_only"):
                        _apply_section_input(parsed_section, clear_mc=True)
                        st.success("Section input loaded from MC_Data A1:S20.")
                else:
                    st.info("Switch to 'Section input (run M-φ)' mode to apply Excel section input.")
            except Exception as exc:
                st.error(str(exc))

    st.subheader("Manual section parameters")
    with st.form("manual_section"):
        cols = st.columns(3)
        manual = dict(st.session_state.section_input)
        fields_per_col = len(SECTION_INPUT_FIELDS) // 3 + 1
        for i, field in enumerate(SECTION_INPUT_FIELDS):
            if field == "code":
                continue
            col = cols[i // fields_per_col]
            default = manual.get(field, DEFAULT_SECTION_VALUES.get(field, 0))
            if field == "n_bar":
                manual[field] = col.number_input(field, value=int(float(default)), step=1)
            else:
                manual[field] = col.number_input(field, value=float(default), format="%.6g")
        manual["code"] = code
        if st.form_submit_button("Use manual values"):
            st.session_state.section_input = manual
            st.session_state.material_curves = sample_material_curves(manual)
            st.success("Manual section input saved.")

with tab_material:
    st.subheader("Stress–Strain curves")
    if st.session_state.section_input and input_mode != "Material curves only":
        if st.button("Generate from material models"):
            st.session_state.material_curves = sample_material_curves(st.session_state.section_input)
    if st.session_state.material_curves is not None:
        fig = plot_stress_strain(st.session_state.material_curves)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Material curve data"):
            st.dataframe(st.session_state.material_curves, use_container_width=True)
    else:
        st.info("Upload section input or material curves CSV to view stress–strain plots.")

with tab_mc:
    st.subheader("Moment–Curvature analysis")
    if input_mode == "Section input (run M-φ)":
        section = compute_derived_geometry(st.session_state.section_input)
        errors = validate_section_input(section)
        if errors:
            st.warning("Section validation:\n" + "\n".join(f"- {e}" for e in errors))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("D (mm)", f"{section['D']:.0f}")
            c2.metric("D0 confined core (mm)", f"{section['D0']:.1f}")
            c3.metric("P range (kN)", f"{section['P_start']:.0f} → {section['P_end']:.0f}")
            if use_excel_pm_grid:
                st.caption(
                    f"Excel PM grid: Pmin={section.get('Pmin', section['P_start']):.0f} … "
                    f"Pmax={section.get('Pmax', section['P_end']):.0f} (60 points)"
                )

        if st.button("Run M-φ analysis", type="primary"):
            progress = st.progress(0, text="Running analysis...")
            status = st.empty()

            def on_progress(done, total):
                progress.progress(done / total, text=f"Computing {done}/{total} points...")

            try:
                with st.spinner("Analyzing..."):
                    results = run_mc_analysis(
                        st.session_state.section_input,
                        n_Pmin=int(n_pmin),
                        n_Pmax=int(n_pmax),
                        n_strain_steps=int(n_strain),
                        progress_callback=on_progress,
                        use_excel_pm_grid=use_excel_pm_grid,
                    )
                st.session_state.mc_results = results_to_dataframe(results)
                st.session_state.pm_results = None
                st.session_state.sap_pm3 = None
                progress.progress(1.0, text="Done")
                status.success(f"Completed {len(results)} points.")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.mc_results is not None:
        mc_df = st.session_state.mc_results
        unique_P = sorted(mc_df["P"].unique())
        selected_P = st.selectbox("Highlight P level", unique_P, index=len(unique_P) // 2)
        st.plotly_chart(plot_mc_family(mc_df, selected_P), use_container_width=True)
        st.dataframe(mc_df.head(100), use_container_width=True)
        st.caption(f"Showing first 100 of {len(mc_df)} rows.")
    else:
        st.info("Run analysis or upload pre-computed M-φ results.")

with tab_pm:
    st.subheader("P–M interaction curve")
    if st.session_state.mc_results is not None:
        section = st.session_state.section_input
        mc_df = st.session_state.mc_results

        c1, c2, c3 = st.columns(3)
        Lp = c1.number_input(
            "Plastic hinge length Lp (m)",
            value=float(section.get("Lp", 1.083)),
            min_value=0.01,
            format="%.4f",
        )
        ec_MD = c2.number_input("ec_MD", value=float(section.get("ec_MD", 0.004973)), format="%.6f")
        ec_CD = c3.number_input("ec_CD", value=float(section.get("ec_CD", 0.008574)), format="%.6f")
        c4, c5, c6 = st.columns(3)
        ec_ED = c4.number_input("ec_ED", value=float(section.get("ec_ED", 0.012862)), format="%.6f")
        hinge_name = c5.text_input("SAP hinge name", value="PM_Hinge")
        auto_compute = c6.checkbox("Auto-compute on tab open", value=True)

        strain_overrides = dict(section)
        strain_overrides.update({"Lp": Lp, "ec_MD": ec_MD, "ec_CD": ec_CD, "ec_ED": ec_ED})

        if auto_compute or st.button("Compute P–M table", type="primary"):
            try:
                if use_excel_pm_grid:
                    P_grid = create_excel_pm_P_array(
                        float(section.get("Pmin", section["P_start"])),
                        float(section.get("Pmax", section["P_end"])),
                        float(section["P_start"]),
                        float(section["P_end"]),
                        int(n_pmin),
                        int(n_pmax),
                    ).tolist()
                else:
                    P_grid = sorted(mc_df["P"].unique())

                pm_df = derive_pm_table(
                    mc_df,
                    strain_limits_from_section(strain_overrides),
                    Lp,
                    P_grid=P_grid,
                )
                st.session_state.pm_results = pm_df
                st.session_state.sap_pm3 = build_sap_pm3(
                    pm_df,
                    hinge_name=hinge_name,
                    Pmin_extreme=float(section.get("Pmin", pm_df.iloc[0]["P"])),
                    Pmax_extreme=float(section.get("Pmax", pm_df.iloc[-1]["P"])),
                )
            except Exception as exc:
                st.error(str(exc))

        if st.session_state.pm_results is not None:
            pm_df = st.session_state.pm_results
            st.plotly_chart(plot_pm_curve(pm_df), use_container_width=True)
            st.dataframe(pm_df, use_container_width=True)

            if st.session_state.pm_reference is not None:
                with st.expander("Compare with uploaded Excel PM summary"):
                    st.dataframe(st.session_state.pm_reference, use_container_width=True)
    else:
        st.info("Complete M-φ analysis or upload an MC workbook first.")

with tab_pushover:
    st.subheader("Derive pushover hinge from M-φ")
    if st.session_state.mc_results is not None:
        mc_df = st.session_state.mc_results
        unique_P = sorted(mc_df["P"].unique())
        P_target = st.select_slider("Axial load P (kN)", options=unique_P, value=unique_P[len(unique_P) // 2])
        if st.button("Derive bilinear hinge", type="primary"):
            try:
                hinge = derive_bilinear_hinge(
                    mc_df, float(P_target), stiffness_ratio=stiffness_ratio
                )
                st.session_state.hinge = hinge
            except Exception as exc:
                st.error(str(exc))

        if st.session_state.hinge:
            h = st.session_state.hinge
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("My (kNm)", f"{h.My:.1f}")
            m2.metric("φy", f"{h.phi_y:.5f}")
            m3.metric("Mu (kNm)", f"{h.Mu:.1f}")
            m4.metric("φu", f"{h.phi_u:.5f}")
            st.plotly_chart(
                plot_mc_single_with_hinge(mc_df, h, float(P_target)),
                use_container_width=True,
            )
            st.subheader("Hinge summary")
            st.dataframe(hinge_to_summary_df(h), use_container_width=True)
            st.subheader("SAP-style backbone points")
            st.dataframe(hinge_to_backbone_df(h), use_container_width=True)
    else:
        st.info("Complete M-φ analysis first.")

with tab_export:
    st.subheader("Download results")
    if st.session_state.mc_results is not None:
        st.download_button(
            "Download M-φ results CSV",
            df_to_csv_bytes(st.session_state.mc_results),
            file_name="mc_results.csv",
            mime="text/csv",
        )
    if st.session_state.hinge:
        h = st.session_state.hinge
        st.download_button(
            "Download hinge summary CSV",
            df_to_csv_bytes(hinge_to_summary_df(h)),
            file_name="pushover_hinge_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download backbone points CSV",
            df_to_csv_bytes(hinge_to_backbone_df(h)),
            file_name="pushover_backbone.csv",
            mime="text/csv",
        )
    if st.session_state.material_curves is not None:
        st.download_button(
            "Download material curves CSV",
            df_to_csv_bytes(st.session_state.material_curves),
            file_name="material_curves.csv",
            mime="text/csv",
        )
    if st.session_state.pm_results is not None:
        st.download_button(
            "Download PM summary CSV",
            df_to_csv_bytes(st.session_state.pm_results),
            file_name="pm_summary.csv",
            mime="text/csv",
        )
    if st.session_state.sap_pm3 is not None:
        st.download_button(
            "Download SAP P-M3 hinge CSV",
            df_to_csv_bytes(st.session_state.sap_pm3),
            file_name="sap_pm3_hinge.csv",
            mime="text/csv",
        )
    if (
        st.session_state.mc_results is None
        and st.session_state.hinge is None
        and st.session_state.material_curves is None
        and st.session_state.pm_results is None
    ):
        st.info("No results to export yet.")
