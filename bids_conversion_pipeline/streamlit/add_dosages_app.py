import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from add_dosages import run_pipeline
from _general_utils import _check_subjects_visits_file
from _streamlit_utils import _select_content

st.set_page_config(layout="centered")

st.title("Add Dosages Pipeline")
st.divider()

st.markdown("""
Pipeline for updating the sessions.tsv files with MPH doses. Useful if this information was not available/still blinded before the subject's data
was converted to BIDS.\n

Example output:

Filename: "sub-101_sessions.tsv"
Contents:
| session_id     | date       | dose |
|----------------|------------|------|
| ses-01         | 01/02/2000 | 0    |
| ses-02         | 03/02/2000 | 10   |

**Note:** For data from unwanted dates, set to a NULL value (leave that cell empty) or exclude that row from the data.\n
""")

st.divider()
st.markdown("**Required Arguments**")

if st.button(
    "Browse for BIDS directory", help="Directory containing the BIDS-compliant dataset."
):
    folder = _select_content("directory")
    if folder:
        st.session_state.bids_dir = folder

    st.session_state.bids_subfolders = sorted(
        [
            x
            for x in Path(st.session_state.bids_dir).glob("*")
            if x.is_dir() and re.match(r"^sub-\d{5}", x.name)
        ]
    )

if st.session_state.get("bids_dir"):
    if st.session_state.get("bids_subfolders"):
        st.success(f"BIDS directory: {st.session_state.bids_dir}")
    else:
        st.error(
            f"Not a valid BIDS directory (no subjects detected): {st.session_state.bids_dir}"
        )

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
        st.session_state.is_valid_visits_file = _check_subjects_visits_file(
            file, dose_column_required=True, for_app=True, return_boolean=True
        )

if st.session_state.get("subjects_visits_file"):
    if st.session_state.is_valid_visits_file:
        st.success(f"Visits File: {st.session_state.subjects_visits_file}")
    else:
        st.error(f"Invalid visits file: {st.session_state.subjects_visits_file} ")

if st.session_state.get("bids_dir") and st.session_state.get("bids_subfolders"):
    st.divider()
    st.markdown("**Optional Arguments**")

    subjects = [
        re.findall(r"\d{5}", x.name)[0]
        for x in st.session_state.get("bids_subfolders")
        if re.findall(r"\d{5}", x.name)
    ]
    subjects = sorted(list(set(subjects)))
    subjects = st.multiselect(
        "Detected subject IDs",
        subjects,
        help="Restrict conversion to specific subjects. Enter IDs without the 'sub-' prefix, separated by commas or spaces.",
    )
else:
    subjects = None

kwargs = {
    "bids_dir": st.session_state.get("bids_dir"),
    "subjects_visits_file": st.session_state.get("subjects_visits_file"),
    "subjects": subjects if subjects else None,
}

st.divider()
if st.button("Run Pipeline", type="primary"):
    if not (
        st.session_state.get("bids_dir") and st.session_state.get("bids_subfolders")
    ):
        st.error("Please select a valid BIDS directory before running.")
    elif not (
        st.session_state.get("subjects_visits_file")
        and st.session_state.get("is_valid_visits_file")
    ):
        st.error("Please upload a valid subjects visits file before running.")
    else:
        with st.spinner("Processing..."):
            run_pipeline(**kwargs)

        st.success("Dosages added to sessions TSV files.")
