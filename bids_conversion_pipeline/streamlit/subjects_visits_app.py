import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "bids_conversion_pipeline")
)

import streamlit as st

from subjects_visits import run_pipeline
from _general_utils import _check_subjects_visits_file
from _streamlit_utils import _select_content

st.set_page_config(layout="centered")

st.title("Subject Visits File Pipeline")
st.divider()
st.markdown("""
    Pipeline for creating the subjects visits file based on the subjects and dates in the raw NIfTI source directory.
    Can also append new subjects and dates to a pre-existing subjects visits file.

    Example output:

    | participant_id | date       |
    |----------------|------------|
    | 101            | 01/02/2000 |
    | 101            | 03/02/2000 |
    | 102            | 01/02/2001 |
    """)

st.divider()

st.markdown("**Required Arguments**")
if st.button(
    "Browse for raw NIfTI directory",
    help="Directory containing the non-BIDS compliant fMRI data.",
):
    folder = _select_content("directory")
    if folder:
        st.session_state.src_dir = folder
        st.session_state.raw_subfolders = sorted(
            [
                x
                for x in Path(st.session_state.src_dir).glob("*")
                if x.is_dir() and re.match(r"^\d{5}", x.name)
            ]
        )

if st.session_state.get("src_dir"):
    if st.session_state.get("raw_subfolders"):
        st.success(f"Source: {st.session_state.src_dir}")
    else:
        st.error(
            f"Not a valid source directory (no subjects detected): {st.session_state.src_dir}"
        )

st.markdown("**Optional Arguments**")
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
        st.session_state.is_valid_visits_file = _check_subjects_visits_file(
            file, dose_column_required=False, for_app=True, return_boolean=True
        )

if st.session_state.get("subjects_visits_file"):
    if st.session_state.is_valid_visits_file:
        st.success(f"Visits File: {st.session_state.subjects_visits_file}")
    else:
        st.error(f"Invalid visits file: {st.session_state.subjects_visits_file} ")

if st.button(
    "Browse for output directory",
    help="Output directory for the subjects visits file.",
):
    folder = _select_content("directory")
    if folder:
        st.session_state.output_dir = folder

if st.session_state.get("output_dir"):
    st.success(f"Output directory: {st.session_state.output_dir}")


kwargs = {
    "src_dir": st.session_state.get("src_dir"),
    "subjects_visits_file": st.session_state.get("subjects_visits_file"),
    "output_dir": st.session_state.get("output_dir"),
}

st.divider()
if st.button("Run Pipeline", type="primary"):
    if not (st.session_state.get("src_dir") and st.session_state.get("raw_subfolders")):
        st.error("Please select a valid raw NIfTI directory before running.")
    elif st.session_state.get("subjects_visits_file") and not st.session_state.get(
        "is_valid_visits_file"
    ):
        st.error("Please select a valid subjects visits file before running.")
    else:
        with st.spinner("Processing..."):
            subjects_visits_file = run_pipeline(**kwargs)

        st.success(f"Subject visits file created/updated at: {subjects_visits_file}")
