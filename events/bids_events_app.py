import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "events"))

import streamlit as st

from create_event_files import run_pipeline

from _streamlit_utils import _select_content

st.title("BIDS Events File App")

st.markdown("**Required Arguments**")

if st.button(
    "Browse for source directory",
    help="Directory containing the log files for the specified task.",
):
    folder = _select_content("directory")
    if folder:
        st.session_state.src_dir = folder

if st.session_state.get("src_dir"):
    st.success(f"Source: {st.session_state.src_dir}")

cohort = st.selectbox(
    "Cohort", ("kids", "adults"), help="Determines which tasks are available."
)

if cohort == "kids":
    valid_tasks = ("nback", "princess", "flanker", "mtle", "mtlr")
else:
    valid_tasks = ("nback", "flanker", "simplegng", "complexgng", "mtle", "mtlr")

task = st.selectbox(
    "Task", valid_tasks, help="The neurobehavioral task to generate event files for."
)

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
    r"%#m/%#d/%Y",
    help=(
        "The date format used in the subjects visits file (e.g., %#m/%#d/%Y). "
        "Note: Excel files may convert dates to %Y-%m-%d regardless of the original format."
    ),
)

st.markdown("**Optional Arguments**")

if st.button("Browse for output directory"):
    folder = _select_content("directory")
    if folder:
        st.session_state.dst_dir = folder

if st.session_state.get("dst_dir"):
    st.success(f"Output: {st.session_state.dst_dir}")

if st.button("Browse for temporary directory"):
    folder = _select_content("directory")
    if folder:
        st.session_state.temp_dir = folder

if st.session_state.get("temp_dir"):
    st.success(f"Temp: {st.session_state.temp_dir}")

subjects = st.text_input(
    "Subject IDs",
    help="Restrict processing to specific subjects. Enter IDs without the 'sub-' prefix, separated by commas or spaces.",
)
if subjects:
    subjects = [s.strip() for s in subjects.replace(",", " ").split() if s.strip()]

minimum_file_size = st.number_input(
    "Minimum file size (bytes)",
    min_value=0,
    value=0,
    help="Files smaller than this are skipped. Leave at 0 to use the task-specific default.",
)

delete_temp_dir = st.checkbox(
    "Delete temporary directory after processing",
    value=True,
    help="If checked, the temporary directory is removed once processing is complete.",
)

exclude_filenames = None
if st.session_state.get("src_dir"):
    subfolders = sorted(
        [x for x in Path(st.session_state.src_dir).glob("*") if x.is_file()]
    )
    exclude_filenames = st.multiselect(
        "Files to exclude",
        subfolders,
        help="Upload any log files that should be excluded from processing.",
    )

kwargs = {
    "src_dir": st.session_state.get("src_dir"),
    "dst_dir": st.session_state.get("dst_dir"),
    "temp_dir": st.session_state.get("temp_dir"),
    "delete_temp_dir": delete_temp_dir,
    "cohort": cohort,
    "task": task,
    "subjects": subjects if subjects else None,
    "subjects_visits_file": st.session_state.get("subjects_visits_file"),
    "subjects_visits_date_fmt": subjects_visits_date_fmt,
    "minimum_file_size": minimum_file_size or None,
    "exclude_filenames": exclude_filenames if exclude_filenames else None,
}

if st.button("Run Pipeline"):
    if not st.session_state.get("src_dir"):
        st.error("Please select a source directory before running.")
    elif not st.session_state.subjects_visits_file:
        st.error("Please upload a subjects visits file before running.")
    else:
        with st.spinner("Processing..."):
            dst_dir = run_pipeline(**kwargs)

        st.success(
            f"Event files for the {task} task ({cohort} cohort) created in: {dst_dir}"
        )
