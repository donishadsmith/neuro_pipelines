import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "bids_conversion_pipeline")
)

import streamlit as st

from add_dosages import run_pipeline
from _streamlit_utils import _select_content

st.title("Add Dosages Pipeline")
st.divider()

st.markdown(
    """**Note:** For data from unwanted dates, set to a NULL value (leave that cell empty) or exclude that row from the data"""
)

st.divider()
st.markdown("**Required Arguments**")

if st.button(
    "Browse for BIDS directory", help="Directory containing the BIDS-compliant dataset."
):
    folder = _select_content("directory")
    if folder:
        st.session_state.bids_dir = folder

if st.session_state.get("bids_dir"):
    st.success(f"BIDS directory: {st.session_state.bids_dir}")

if st.button(
    "Browse for subjects visits file",
    help=(
        "A CSV or Excel file mapping subjects to visit dates and dosages. "
        "Must contain 'participant_id', 'date', and 'dose' columns. "
        "List dates in chronological order per subject and use NaN for missing sessions. "
        "Do not include unwanted subject dates in order to skip them. "
        "If `dose_mg` (only relevant to adult cohort since the dose column is coded as 'mph' and 'placebo') "
        "is a column in the file, then that information will be included too."
    ),
):
    file = _select_content("file")
    if file:
        st.session_state.subjects_visits_file = file

if st.session_state.get("subjects_visits_file"):
    st.success(f"Visits File: {st.session_state.subjects_visits_file}")

st.divider()
st.markdown("**Optional Arguments**")

subjects = st.text_input(
    "Subject IDs",
    help="Restrict processing to specific subjects. Enter IDs without the 'sub-' prefix, separated by commas or spaces.",
)
if subjects:
    subjects = [s.strip() for s in subjects.replace(",", " ").split() if s.strip()]

kwargs = {
    "bids_dir": st.session_state.get("bids_dir"),
    "subjects_visits_file": st.session_state.get("subjects_visits_file"),
    "subjects": subjects if subjects else None,
}

st.divider()
if st.button("Run Pipeline", type="primary"):
    if not st.session_state.get("bids_dir"):
        st.error("Please select a BIDS directory before running.")
    elif not st.session_state.subjects_visits_file:
        st.error("Please upload a subjects visits file before running.")
    else:
        with st.spinner("Processing..."):
            run_pipeline(**kwargs)

        st.success("Dosages added to sessions TSV files.")
