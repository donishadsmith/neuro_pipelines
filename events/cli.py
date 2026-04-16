import argparse

from create_event_files import run_pipeline


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Create BIDS-compliant event files from neurobehavioral log data."
    )
    parser.add_argument(
        "--src_dir",
        dest="src_dir",
        required=True,
        help="Directory containing the neurobehavioral log files.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=False,
        default=None,
        help="Output directory for the generated event files. Defaults to ~/BIDS_Events.",
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
            "Dates should be listed in chronological order per subject. "
            "Use NaN for missing sessions."
        ),
    )
    parser.add_argument(
        "--subjects_visits_date_fmt",
        dest="subjects_visits_date_fmt",
        required=False,
        default=r"%#m/%#d/%Y",
        help=(
            "Date format used in the subjects visits file (e.g., '%%#m/%%#d/%%Y'). "
            "Note: Excel files may convert dates to '%%Y-%%m-%%d' regardless of the original format."
        ),
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


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    run_pipeline(**vars(args))
