import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "merge_data"))

import streamlit as st

from merge_data import run_pipeline
from _streamlit_utils import _select_content


from _general_utils import _get_dataframe

st.set_page_config(layout="centered")

st.title("Merge Data Pipeline")
st.divider()

st.markdown("""
Pipeline for merging data from a primary and secondary file.

**Notes:**
- Rows are always matched on subject ID. Date and dose are optional and only used for matching
when a column is selected for both files.
- For data that does not vary by visit (e.g., demographics), only select the subject ID columns.
The values will be copied to every row for that subject.
- For data that varies by visit (e.g., beta coefficients), also select the date columns.
- If a file only contains specific doses for subjects (e.g., "0_vs_10", "10_vs_0", "5_vs_0", etc), then
it is recommended to also select the dose columns for both the primary and secondary files
- You can keep updating the secondary file to merge different files with the primary file.
- Columns in the secondary file that share a name with a column already in the primary file cannot be added
as is, to prevent overwriting or duplicating data. Use the optional "Suffix for added columns" field to append
a unique suffix to every added column (e.g., "GLM_Individual_Cluster_Beta" -> "GLM_Individual_Cluster_Beta_5_vs_0_superior_parietal"),
or rename the columns in the secondary file before browsing for it
""")

st.divider()
st.markdown("**Required Arguments**")
if st.button(
    "Browse for primary file",
    help="Path to file to use as the base file to receive new columns.",
):
    file = _select_content("file")
    if file:
        st.session_state.primary_file = file
        st.session_state.primary_columns = _get_dataframe(file).columns.tolist()

if st.session_state.get("primary_file"):
    st.success(f"Primary file: {st.session_state.primary_file}")

if st.session_state.get("primary_file"):
    st.session_state.primary_file_subject_column = st.selectbox(
        "Primary File Subject ID Column:",
        st.session_state.primary_columns,
        index=None,
        help="Name of the column containing the subject IDs in the primary file.",
    )

if st.button(
    "Browse for secondary file",
    help="Path to file containing the new columns to add to the primary file.",
):
    file = _select_content("file")
    if file:
        st.session_state.secondary_file = file

if st.session_state.get("secondary_file"):
    st.success(f"Secondary file: {st.session_state.secondary_file}")

if filename := st.session_state.get("secondary_file"):
    secondary_columns = _get_dataframe(filename).columns.tolist()
    st.session_state.secondary_file_subject_column = st.selectbox(
        "Secondary File Subject ID Column:",
        secondary_columns,
        index=None,
        help="Name of the column containing the subject IDs in the secondary file.",
    )

    secondary_columns = [
        col
        for col in secondary_columns
        if col != st.session_state.secondary_file_subject_column
    ]

    st.session_state.columns_to_add = st.multiselect(
        "Columns to add:",
        secondary_columns,
        help=(
            "Column names from the secondary file to add to the primary file. "
            "Columns that already exist in the primary file need a suffix (see Optional Arguments)."
        ),
    )

if st.session_state.get("primary_file") or st.session_state.get("secondary_file"):
    st.divider()
    st.markdown("**Optional Arguments**")
    if st.session_state.get("primary_file"):
        primary_columns = [
            col
            for col in st.session_state.primary_columns
            if col != st.session_state.primary_file_subject_column
        ]

        st.session_state.primary_file_date_column = st.selectbox(
            "Primary File Date Column:",
            primary_columns,
            index=None,
            help=(
                "Name of the column containing the dates in the primary file. "
                "Leave empty for data that does not vary by visit."
            ),
        )

        primary_columns = [
            col
            for col in primary_columns
            if col != st.session_state.primary_file_date_column
        ]

        st.session_state.primary_file_dose_column = st.selectbox(
            "Primary File Dose Column:",
            primary_columns,
            index=None,
            help="Name of the column containing the dose in the primary file.",
        )

    if filename := st.session_state.get("secondary_file"):
        secondary_columns = _get_dataframe(filename).columns.tolist()
        secondary_columns = [
            col
            for col in secondary_columns
            if col
            not in [
                st.session_state.secondary_file_subject_column,
                *(st.session_state.columns_to_add or []),
            ]
        ]

        st.session_state.secondary_file_date_column = st.selectbox(
            "Secondary File Date Column:",
            secondary_columns,
            index=None,
            help=(
                "Name of the column containing the dates in the secondary file. "
                "Leave empty for data that does not vary by visit."
            ),
        )

        secondary_columns = [
            col
            for col in secondary_columns
            if col != st.session_state.secondary_file_date_column
        ]

        st.session_state.secondary_file_dose_column = st.selectbox(
            "Secondary File Dose Column:",
            secondary_columns,
            index=None,
            help="Name of the column containing the dose in the secondary file.",
        )

        st.session_state.column_suffix = st.text_input(
            "Suffix for added columns:",
            placeholder="e.g., _5_vs_0_superior_parietal",
            help=(
                "Appended to every column selected in 'Columns to add'. "
                "Required if any of those columns already exist in the primary file."
            ),
        ).strip()


kwargs = {
    "primary_file": st.session_state.get("primary_file"),
    "secondary_file": st.session_state.get("secondary_file"),
    "primary_file_subject_column": st.session_state.get("primary_file_subject_column"),
    "primary_file_date_column": st.session_state.get("primary_file_date_column"),
    "secondary_file_subject_column": st.session_state.get(
        "secondary_file_subject_column"
    ),
    "secondary_file_date_column": st.session_state.get("secondary_file_date_column"),
    "columns_to_add": st.session_state.get("columns_to_add"),
    "primary_file_dose_column": st.session_state.get("primary_file_dose_column"),
    "secondary_file_dose_column": st.session_state.get("secondary_file_dose_column"),
    "column_suffix": st.session_state.get("column_suffix") or None,
}

st.divider()
if st.button("Run Pipeline", type="primary"):

    suffix = st.session_state.get("column_suffix") or ""
    existing_columns = (
        _get_dataframe(st.session_state.primary_file).columns.tolist()
        if st.session_state.get("primary_file")
        else []
    )
    conflicting_columns = [
        f"{col}{suffix}"
        for col in (st.session_state.get("columns_to_add") or [])
        if f"{col}{suffix}" in existing_columns
    ]

    if not st.session_state.get("primary_file"):
        st.error("Please select a primary file before running.")
    elif not st.session_state.get("secondary_file"):
        st.error("Please select a secondary file before running.")
    elif not st.session_state.get("primary_file_subject_column"):
        st.error("Please select a primary subject column before running.")
    elif not st.session_state.get("secondary_file_subject_column"):
        st.error("Please select a secondary subject column before running.")
    elif not st.session_state.get("columns_to_add"):
        st.error("Please select covariates to add before running.")
    elif (st.session_state.get("primary_file_date_column") is None) != (
        st.session_state.get("secondary_file_date_column") is None
    ):
        st.error(
            "Please select the date columns for both the primary and secondary file if merging should also consider date."
        )
    elif (st.session_state.get("primary_file_dose_column") is None) != (
        st.session_state.get("secondary_file_dose_column") is None
    ):
        st.error(
            "Please select the dose columns for both the primary and secondary file if merging should also consider dose."
        )
    elif conflicting_columns:
        st.error(
            "The following columns already exist in the primary file: "
            f"{', '.join(conflicting_columns)}. Add or change the suffix for added "
            "columns, or rename them in the secondary file."
        )
    else:
        with st.spinner("Processing..."):
            new_columns = run_pipeline(**kwargs)

        st.success(f"Columns added to: {st.session_state.get('primary_file')}")

        if new_columns:
            st.info(
                f"Names of new columns added to primary file: {', '.join(new_columns)}"
            )
        else:
            st.info(
                "No new columns added due to column names already being present in the primary file."
            )
