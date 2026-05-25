import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "afni_whereami"))

import streamlit as st
from whereami import run_pipeline

st.title("AFNI WhereAmI Pipeline")
st.divider()

st.markdown("Pipeline for looking up MNI coordinates with AFNI's ``whereami``.")

st.divider()

st.markdown("**Required Arguments**")

cohort = st.selectbox("Cohort", ("kids", "adults"), help="Determines the atlas used.")

col1, col2, col3 = st.columns(3)

with col1:
    X = st.number_input("X", help="X coordinate in MNI space.", format="%d", value=0)

with col2:
    Y = st.number_input("Y", help="Y coordinate in MNI space.", format="%d", value=0)

with col3:
    Z = st.number_input("Z", help="Z coordinate in MNI space.", format="%d", value=0)

kwargs = {
    "cohort": cohort,
    "mni_coordinate": [X, Y, Z],
}

st.divider()

if st.button("Run Pipeline", type="primary"):
    with st.spinner("Processing..."):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text(run_pipeline(**kwargs), text_alignment="center")
