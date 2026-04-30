import argparse, logging
from pathlib import Path

import streamlit as st

from _streamlit_utils import StreamlitLogHandler, _select_content


def _get_cmd_args(caller):
    parser = argparse.ArgumentParser(
        description="Create BIDS-compliant event files from neurobehavioral log data."
    )
    parser.add_argument(
        "--log_dir",
        dest="log_dir",
        required=True,
        help="Directory containing the neurobehavioral log files.",
    )

    default_dir = "~/BIDS_Events" if caller == "BIDS Events" else "~/Behavioral_Data."
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=False,
        default=None,
        help=f"Output directory for the generated event files. Defaults to {default_dir}",
    )
    parser.add_argument(
        "--temp_dir",
        dest="temp_dir",
        required=False,
        default=None,
        help="Temporary working directory used during processing. Cleaned up automatically after completion.",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=False,
        default="kids",
        choices=["kids", "adults"],
        help="Cohort name. Determines which tasks are available. Default: kids.",
    )
    parser.add_argument(
        "--task",
        dest="task",
        required=True,
        help=(
            "Task name. Kids: nback, flanker, mtle, mtlr, princess. "
            "Adults: nback, flanker, simplegng, complexgng, mtle, mtlr."
        ),
    )
    parser.add_argument(
        "--subjects",
        dest="subjects",
        required=False,
        default=None,
        nargs="+",
        help="One or more subject IDs (without the 'sub-' prefix) to restrict processing to.",
    )
    parser.add_argument(
        "--minimum_file_size",
        dest="minimum_file_size",
        required=False,
        default=None,
        help="Minimum file size in bytes. Files smaller than this are skipped. Uses task-specific defaults if not set.",
    )
    parser.add_argument(
        "--subjects_visits_file",
        dest="subjects_visits_file",
        required=True,
        type=str,
        help=(
            "Path to a CSV or Excel file mapping subjects to visit dates. "
            "Must contain 'participant_id' and 'date' columns. "
            "Ensure all dates use the same format. "
            "For data from unwanted dates, set to a null value (leave that cell empty) or exclude that row from the data."
        ),
    )

    if caller == "Behavioral Data":
        parser.add_argument(
            "--behavioral_data_file",
            dest="behavioral_data_file",
            required=False,
            default=None,
            help="The path to the behavioral data (if exists) to append new data to.",
        )

    parser.add_argument(
        "--delete_temp_dir",
        dest="delete_temp_dir",
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Delete the temporary directory after processing. Default: True.",
    )
    parser.add_argument(
        "--exclude_filenames",
        dest="exclude_filenames",
        required=False,
        default=None,
        nargs="+",
        help="Filenames to exclude from processing (e.g., 101_nback.txt 102_flanker.xls).",
    )

    return parser


def _app(caller, pipeline):
    if caller == "BIDS Events":
        st.title("BIDS Events File Pipeline")
        note = (
            "**Note:**\n"
            "- For data from unwanted dates, set to a NULL value (leave that cell empty) or exclude that row from the data"
        )
    else:
        st.title("Behavioral Data Pipeline")
        note = ""

    st.divider()

    st.markdown(note)

    if caller == "BIDS Events":
        st.markdown(
            "\n**Use the 'Move Files Pipeline' to automatically move events TSV files to thier respective subjects directory within the BIDS directory.**"
        )

    st.divider()
    st.markdown("**Required Arguments**")

    if st.button(
        "Browse for log directory",
        help="Directory containing the log files for the specified task.",
    ):
        folder = _select_content("directory")
        if folder:
            st.session_state.log_dir = folder

    if st.session_state.get("log_dir"):
        st.success(f"Source: {st.session_state.log_dir}")

    cohort = st.selectbox(
        "Cohort", ("kids", "adults"), help="Determines which tasks are available."
    )

    if cohort == "kids":
        valid_tasks = ("nback", "princess", "flanker", "mtle", "mtlr")
    else:
        valid_tasks = ("nback", "flanker", "simplegng", "complexgng", "mtle", "mtlr")

    task = st.selectbox(
        "Task",
        valid_tasks,
        help="The neurobehavioral task to generate event files for.",
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

    st.divider()
    st.markdown("**Optional Arguments**")

    if st.button("Browse for output directory"):
        folder = _select_content("directory")
        if folder:
            st.session_state.dst_dir = folder

    if st.session_state.get("dst_dir"):
        st.success(f"Output: {st.session_state.dst_dir}")

    if caller == "Behavioral Data" and st.button(
        "Browse behavioral data file",
        help="The path to the behavioral data (if exists) to append new data to.",
    ):
        file = _select_content("file")
        if file:
            st.session_state.behavioral_data_file = file

    if st.session_state.get("behavioral_data_file"):
        st.success(f"Behavioral File: {st.session_state.behavioral_data_file}")

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
    if st.session_state.get("log_dir"):
        subfolders = sorted(
            [x for x in Path(st.session_state.log_dir).glob("*") if x.is_file()]
        )
        exclude_filenames = st.multiselect(
            "Files to exclude",
            subfolders,
            help="Upload any log files that should be excluded from processing.",
        )

    kwargs = {
        "log_dir": st.session_state.get("log_dir"),
        "dst_dir": st.session_state.get("dst_dir"),
        "temp_dir": st.session_state.get("temp_dir"),
        "delete_temp_dir": delete_temp_dir,
        "cohort": cohort,
        "task": task,
        "subjects": subjects if subjects else None,
        "subjects_visits_file": st.session_state.get("subjects_visits_file"),
        "behavioral_data_file": st.session_state.get("behavioral_data_file"),
        "minimum_file_size": minimum_file_size or None,
        "exclude_filenames": exclude_filenames if exclude_filenames else None,
    }
    if caller == "BIDS Events":
        del kwargs["behavioral_data_file"]

    st.divider()
    if st.button("Run Pipeline", type="primary"):
        if not st.session_state.get("log_dir"):
            st.error("Please select a log directory before running.")
        elif not st.session_state.subjects_visits_file:
            st.error("Please upload a subjects visits file before running.")
        else:
            status_container = st.empty()
            with status_container.status(
                "Running pipeline...", expanded=True
            ) as status:

                handler = StreamlitLogHandler(status)
                logging.getLogger().addHandler(handler)

                dst_dir = pipeline(**kwargs, caller=caller)

                logging.getLogger().removeHandler(handler)

                file_type_str = (
                    "Event files" if caller == "BIDS Events" else "Behavioral CSV"
                )
                status.update(
                    label=f"{file_type_str} for the {task} task ({cohort} cohort) created in: {dst_dir}",
                    state="complete",
                    expanded=False,
                )
