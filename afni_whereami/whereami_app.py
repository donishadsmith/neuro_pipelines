import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "afni_whereami"))

import streamlit as st
from bidsaid._helpers import iterable_to_str
from whereami import run_pipeline

st.title("AFNI WhereAmI Pipeline")
st.divider()

st.markdown(
    "Pipeline for looking up MNI coordinates with AFNI's ``whereami``. "
    "Uses the Freesurfer MNI2009c DK parcellation and the "
    "Brodmann atlas for MNI 2009c - Pijnenburg AFNI version."
)

st.divider()

st.markdown("**Required Arguments**")

cohort = st.selectbox(
    "Cohort",
    ("kids", "adults"),
    help="Determines the template space used for visualization purposes.",
)

col1, col2, col3 = st.columns(3)

with col1:
    X = st.number_input("X", help="X coordinate in MNI space.", format="%d", value=0)

with col2:
    Y = st.number_input("Y", help="Y coordinate in MNI space.", format="%d", value=0)

with col3:
    Z = st.number_input("Z", help="Z coordinate in MNI space.", format="%d", value=0)

st.divider()

kwargs = {"cohort": cohort, "mni_coordinate": [X, Y, Z]}

if st.button("Run Pipeline", type="primary"):
    with st.spinner("Processing..."):
        output_text, display, sphere_radius_text = run_pipeline(**kwargs)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text(output_text, text_alignment="center")

        if display:
            st.pyplot(display)
            additional_text = (
                f"[{iterable_to_str([X, Y, Z])}]"
                if "focus point" in sphere_radius_text.lower()
                else f"from {iterable_to_str([X, Y, Z])}"
            )
            st.caption(
                "**Note:** The radius of the sphere mask (red) adjusts to match the distance of the closest "
                f"label from the Freesurfer MNI2009c DK parcellation (which is {sphere_radius_text} {additional_text})."
            )
