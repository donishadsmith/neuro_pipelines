import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bids_conversion_pipeline"))

import streamlit as st

from participants_tsv import run_pipeline
from _streamlit_utils import _select_content

st.title("Participants TSV")

st.markdown("**Required Arguments**")

if st.button(
    "Browse for BIDS directory", help="Directory containing the BIDS-compliant dataset."
):
    folder = _select_content("directory")
    if folder:
        st.session_state.bids_dir = folder

if st.session_state.get("bids_dir"):
    st.success(f"BIDS directory: {st.session_state.bids_dir}")

st.markdown("**Optional Arguments**")

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
    "bids_dir": st.session_state.get("bids_dir"),
    "demographics_file": st.session_state.get("demographics_file"),
    "covariates_to_add": covariates_to_add if covariates_to_add else None,
}

if st.button("Run Pipeline"):
    if not st.session_state.get("bids_dir"):
        st.error("Please select a BIDS directory before running.")
    else:
        with st.spinner("Processing..."):
            run_pipeline(**kwargs)

        st.success("Participants TSV updated.")
