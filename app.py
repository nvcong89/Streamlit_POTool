"""Push Over Analysis — Streamlit home page."""

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Push Over Analysis Tool",
    page_icon="📊",
    layout="wide",
)

st.title("Push Over Analysis Tool")
st.markdown(
    """
    Web-based **moment–curvature** and **pushover hinge** analysis for pile-plug sections.

    ### Workflow
    1. **Upload** section parameters or pre-computed M–φ results (CSV/Excel)
    2. **Review** stress–strain material curves
    3. **Run** or load moment–curvature analysis
    4. **Derive** bilinear pushover hinge backbone from M–φ
    5. **Export** results as CSV

    Open **Pushover Analysis** in the sidebar to start.
    """
)

templates_dir = Path(__file__).parent / "templates"
st.subheader("Templates")
col1, col2 = st.columns(2)
with col1:
    section_tpl = (templates_dir / "section_input.csv").read_text(encoding="utf-8")
    st.download_button(
        "Download section input template",
        section_tpl,
        file_name="section_input.csv",
        mime="text/csv",
    )
with col2:
    mc_tpl = (templates_dir / "mc_results.csv").read_text(encoding="utf-8")
    st.download_button(
        "Download M–φ results template",
        mc_tpl,
        file_name="mc_results.csv",
        mime="text/csv",
    )

st.info("Units: mm, MPa, kN, kNm. Compatible with CN PushOver Excel tool output format.")
