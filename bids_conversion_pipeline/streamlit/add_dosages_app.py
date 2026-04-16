import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bids_conversion_pipeline"))

import streamlit as st

from add_dosages import run_pipeline
from _streamlit_utils import _select_content

st.title("Add Dosages")

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
        "Do not include unwanted subject dates in order to skip them."
    ),
):
    file = _select_content("file")
    if file:
        st.session_state.subjects_visits_file = file

if st.session_state.get("subjects_visits_file"):
    st.success(f"Visits File: {st.session_state.subjects_visits_file}")


subjects_visits_date_fmt = st.text_input(
    "Date format in the subjects visits file",
    r"%m/%d/%Y",
    help="The date format used in the subjects visits file (e.g., %m/%d/%Y).",
)

sessions_tsv_date_fmt = st.text_input(
    "Date format in the sessions TSV files",
    r"%y%m%d",
    help="The date format used in existing sessions TSV files (e.g., %y%m%d).",
)

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
    "subjects_visits_date_fmt": subjects_visits_date_fmt,
    "sessions_tsv_date_fmt": sessions_tsv_date_fmt,
}

if st.button("Run Pipeline"):
    if not st.session_state.get("bids_dir"):
        st.error("Please select a BIDS directory before running.")
    elif not st.session_state.subjects_visits_file:
        st.error("Please upload a subjects visits file before running.")
    else:
        with st.spinner("Processing..."):
            run_pipeline(**kwargs)

        st.success("Dosages added to sessions TSV files.")
