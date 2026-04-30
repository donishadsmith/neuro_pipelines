import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "bids_conversion_pipeline")
)

import streamlit as st

from participants_demographics import run_pipeline
from _streamlit_utils import _select_content

st.title("Participants Demographics Pipeline")
st.divider()

st.markdown("""**Note:**\n
- The demographics data file must have a column named "participant_id".
""")

st.divider()
st.markdown("**Required Arguments**")
if st.button(
    "Browse for participants TSV file",
    help="A BIDS compliant participants TSV ('participants.tsv') file.",
):
    file = _select_content("file")
    if file:
        st.session_state.participants_tsv_path = file

if st.session_state.get("participants_tsv_path"):
    st.success(f"Participants TSV file: {st.session_state.participants_tsv_path}")

if st.button(
    "Browse for demographics file",
    help="A CSV or Excel file containing subject demographics. Must have a 'participant_id' column.",
):
    file = _select_content("file")
    if file:
        st.session_state.demographics_file = file

if st.session_state.get("demographics_file"):
    st.success(f"Demographics file: {st.session_state.demographics_file}")

covariates_to_add = st.text_input(
    "Covariates to add",
    help="Column names from the demographics file to include in the participants TSV. Separated by commas or spaces.",
)
if covariates_to_add:
    covariates_to_add = [
        s.strip() for s in covariates_to_add.replace(",", " ").split() if s.strip()
    ]

kwargs = {
    "participants_tsv_path": st.session_state.get("participants_tsv_path"),
    "demographics_file": st.session_state.get("demographics_file"),
    "covariates_to_add": covariates_to_add,
}

st.divider()
if st.button("Run Pipeline", type="primary"):
    if not st.session_state.get("participants_tsv_path"):
        st.error("Please select a participants TSV file before running.")
    elif not st.session_state.get("demographics_file"):
        st.error("Please select a demographics file before running.")
    elif not covariates_to_add:
        st.error(
            "Please select add the names of the demographic covariates before running."
        )
    else:
        with st.spinner("Processing..."):
            run_pipeline(**kwargs)

        st.success(
            f"Demographic data added/updated to: {st.session_state.get('participants_tsv_path')}."
        )
