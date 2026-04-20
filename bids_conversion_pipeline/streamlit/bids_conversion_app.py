import logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "bids_conversion_pipeline")
)

import streamlit as st

from bids_conversion import run_pipeline
from _streamlit_utils import StreamlitLogHandler, _select_content

st.title("NIfTI to BIDS Conversion Pipeline")
st.divider()

st.markdown("""**Note:**\n
- Run 'Participants TSV Pipeline' and 'Add Dosages Pipeline' after conversion.\n
- If the BIDS directory has a participants TSV file, it will not be overwritten, the new subjects will be appended.\n

**Date Format Cheatsheet:**\n

**% is a placeholder prefix and must be included when inputing the date format for the subject visits CSV or the source**
**directory folders (copy and paste the relevant format after ->)**

- 2025-01-02 -> %Y-%m-%d
- 01/02/2025 -> %m/%d/%Y or %#m/%#d/%Y
- 01/02/25 -> %m/%d/%y
- 20250102 -> %y%m%d

**If the subjects visits CSV is an Excel file (.xlsx extension), use the following date format: %Y-%m-%d**
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

if st.session_state.get("src_dir"):
    st.success(f"Source: {st.session_state.src_dir}")

cohort = st.selectbox(
    "Cohort",
    ("kids", "adults"),
    help="Determines which scan protocols and tasks are available.",
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
    r"%m/%d/%Y",
    help=(
        "The date format used in the subjects visits file (e.g., %m/%d/%Y). "
        "Note: Excel files may convert dates to %Y-%m-%d regardless of the original format."
    ),
)
subjects_visits_date_fmt = rf"{subjects_visits_date_fmt.strip()}"

src_data_date_fmt = st.text_input(
    "Date format in the source folder names",
    r"%y%m%d",
    help="The date format used in the source directory folder names (e.g., %y%m%d).",
)
src_data_date_fmt = rf"{src_data_date_fmt.strip()}"

st.divider()
st.markdown("**Optional Arguments**")

if st.button("Browse for BIDS output directory"):
    folder = _select_content("directory")
    if folder:
        st.session_state.bids_dir = folder

if st.session_state.get("bids_dir"):
    st.success(f"BIDS output: {st.session_state.bids_dir}")

if st.button("Browse for temporary directory"):
    folder = _select_content("directory")
    if folder:
        st.session_state.temp_dir = folder

if st.session_state.get("temp_dir"):
    st.success(f"Temp: {st.session_state.temp_dir}")

subjects = st.text_input(
    "Subject IDs",
    help="Restrict conversion to specific subjects. Enter IDs without the 'sub-' prefix, separated by commas or spaces.",
)
if subjects:
    subjects = [s.strip() for s in subjects.replace(",", " ").split() if s.strip()]

exclude_src_folder_names = None
if st.session_state.get("src_dir"):
    subfolders = sorted(
        [x for x in Path(st.session_state.src_dir).glob("*") if x.is_dir()]
    )
    exclude_src_folder_names = st.multiselect(
        "Source folders to exclude",
        subfolders,
        help="Select any source folders to skip during conversion.",
    )

exclude_nifti_filenames = None
if st.session_state.get("src_dir"):
    nifti_files = sorted(
        [
            x
            for x in Path(st.session_state.src_dir).rglob("*")
            if x.is_file() and str(x).endswith((".nii", ".nii.gz"))
        ]
    )
    exclude_nifti_filenames = st.multiselect(
        "NIfTI filenames to exclude",
        nifti_files,
        help="Select any NIfTI files to skip during conversion.",
    )

delete_temp_dir = st.checkbox(
    "Delete temporary directory after processing",
    value=True,
    help="If checked, the temporary directory is removed once processing is complete.",
)

create_dataset_metadata = st.checkbox(
    "Create dataset metadata",
    value=False,
    help=(
        "Create the participants TSV and dataset description JSON. "
        "Appends to an existing participants TSV if one is found. "
        "Keep unchecked when running in parallel to avoid race conditions."
    ),
)

add_sessions_tsv = st.checkbox(
    "Add sessions TSV",
    value=False,
    help="Create a sessions TSV file containing the session and scan date for each subject.",
)

kwargs = {
    "src_dir": st.session_state.get("src_dir"),
    "temp_dir": st.session_state.get("temp_dir"),
    "bids_dir": st.session_state.get("bids_dir"),
    "subjects": subjects if subjects else None,
    "exclude_src_folder_names": (
        exclude_src_folder_names if exclude_src_folder_names else None
    ),
    "exclude_nifti_filenames": (
        exclude_nifti_filenames if exclude_nifti_filenames else None
    ),
    "cohort": cohort,
    "delete_temp_dir": delete_temp_dir,
    "create_dataset_metadata": create_dataset_metadata,
    "add_sessions_tsv": add_sessions_tsv,
    "subjects_visits_file": st.session_state.get("subjects_visits_file"),
    "subjects_visits_date_fmt": subjects_visits_date_fmt,
    "src_data_date_fmt": src_data_date_fmt,
}

st.divider()
if st.button("Run Pipeline", type="primary"):
    if not st.session_state.get("src_dir"):
        st.error("Please select a source directory before running.")
    elif not st.session_state.subjects_visits_file:
        st.error("Please upload a subjects visits file before running.")
    else:
        status_container = st.empty()
        with status_container.status("Running pipeline...", expanded=True) as status:
            handler = StreamlitLogHandler(status)
            logging.getLogger().addHandler(handler)

            bids_dir = run_pipeline(**kwargs)

            logging.getLogger().removeHandler(handler)

            status.update(
                label=f"BIDS conversion complete. Dataset located at {bids_dir}",
                state="complete",
                expanded=False,
            )
