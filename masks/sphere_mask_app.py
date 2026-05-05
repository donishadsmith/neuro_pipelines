import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "masks"))

import streamlit as st

from _streamlit_utils import _select_content
from sphere_mask import run_pipeline

st.title("Sphere Mask Pipeline")
st.divider()

st.markdown(
    "Pipeline for generating a-priori sphere masks for seed-based connectivity analyses."
)

st.divider()

st.markdown("**Required Arguments**")

cohort = st.selectbox(
    "Cohort", ("kids", "adults"), help="Determines the template space used."
)

col1, col2, col3 = st.columns(3)

with col1:
    X = st.number_input("X", help="MNI X coordinate.", format="%d", value=0)

with col2:
    Y = st.number_input("Y", help="MNI Y coordinate.", format="%d", value=0)

with col3:
    Z = st.number_input("Z", help="MNI Z coordinate.", format="%d", value=0)

sphere_radius = st.number_input(
    "Sphere radius",
    help="The radius of the sphere mask in mm.",
    min_value=3,
    max_value=10,
    format="%d",
)
st.divider()

st.markdown("**Optional Arguments**")

if st.button("Browse for output directory"):
    folder = _select_content("directory")
    if folder:
        st.session_state.sphere_mask_dst_dir = folder

if st.session_state.get("sphere_mask_dst_dir"):
    st.success(f"Output directory: {st.session_state.sphere_mask_dst_dir}")

kwargs = {
    "cohort": cohort,
    "mni_coordinate": [X, Y, Z],
    "sphere_radius": sphere_radius,
    "dst_dir": st.session_state.get("sphere_mask_dst_dir"),
}

st.divider()

if st.button("Run Pipeline", type="primary"):
    with st.spinner("Processing..."):
        sphere_filename, plot_filename = run_pipeline(**kwargs)

    st.success(f"Sphere mask created at: {sphere_filename}")
    st.success(f"Sphere plot created at: {plot_filename}")

    st.markdown("Sphere Mask (Red)")

    st.image(plot_filename)
