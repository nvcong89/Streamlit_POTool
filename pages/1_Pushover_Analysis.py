"""Pushover Analysis — main Streamlit tool page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.mc_runner import run_mc_analysis, results_to_dataframe
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
    parse_section_input,
    template_mc_results_csv,
    template_section_kv_csv,
)
from src.ui.charts import plot_mc_family, plot_mc_single_with_hinge, plot_stress_strain

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

    st.subheader("Hinge idealization")
    stiffness_ratio = st.slider(
        "Yield stiffness ratio (fraction of K0)",
        min_value=0.05,
        max_value=0.40,
        value=0.15,
        step=0.01,
    )

tab_upload, tab_material, tab_mc, tab_pushover, tab_export = st.tabs(
    ["Upload & Map", "Material", "Moment–Curvature", "Pushover Derivation", "Export"]
)

with tab_upload:
    st.subheader("Upload data")
    uploaded = st.file_uploader(
        "CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Use templates from the home page or your own column names with the mapper below.",
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
                    st.success(f"Loaded {len(mc_df)} M-φ points.")
                else:
                    mat_df = parse_material_curves(raw_df, rename_for_parse or None)
                    st.session_state.material_curves = mat_df
                    st.success(f"Loaded {len(mat_df)} material curve points.")

            with st.expander("Preview raw data"):
                st.dataframe(raw_df.head(20), use_container_width=True)

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
                    )
                st.session_state.mc_results = results_to_dataframe(results)
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
    if (
        st.session_state.mc_results is None
        and st.session_state.hinge is None
        and st.session_state.material_curves is None
    ):
        st.info("No results to export yet.")
