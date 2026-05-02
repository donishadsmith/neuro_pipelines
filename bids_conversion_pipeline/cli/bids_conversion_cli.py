import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bids_conversion import run_pipeline
from _general_utils import _convert_to_bool


def _get_cmd_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a dataset to BIDS format.")
    parser.add_argument(
        "--src_dir",
        dest="src_dir",
        required=True,
        help=(
            "Source directory containing the original data. "
            "NIfTI files should be stored in folders named {participant_id}_{date}."
        ),
    )
    parser.add_argument(
        "--temp_dir",
        dest="temp_dir",
        required=False,
        default=None,
        help="Temporary working directory for intermediate files.",
    )
    parser.add_argument(
        "--bids_dir",
        dest="bids_dir",
        required=False,
        default=None,
        help="Output BIDS directory. Defaults to ~/BIDS_Events.",
    )
    parser.add_argument(
        "--subjects",
        dest="subjects",
        required=False,
        nargs="+",
        default=None,
        help="One or more subject IDs (without the 'sub-' prefix) to restrict conversion to.",
    )
    parser.add_argument(
        "--exclude_src_folder_names",
        dest="exclude_src_folder_names",
        required=False,
        nargs="+",
        default=None,
        help="Source folder names to exclude (e.g., 101_1111).",
    )
    parser.add_argument(
        "--exclude_nifti_filenames",
        dest="exclude_nifti_filenames",
        required=False,
        nargs="+",
        default=None,
        help="NIfTI filenames to exclude (e.g., 101_4444.nii).",
    )
    parser.add_argument(
        "--delete_temp_dir",
        dest="delete_temp_dir",
        required=False,
        default=True,
        type=_convert_to_bool,
        help="Delete the temporary directory after processing. Default: True.",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=False,
        default="kids",
        choices=["kids", "adults"],
        help="Cohort name. Default: kids.",
    )
    parser.add_argument(
        "--create_dataset_metadata",
        dest="create_dataset_metadata",
        required=False,
        default=True,
        type=_convert_to_bool,
        help=(
            "Create the participants TSV and dataset description JSON. "
            "Appends to an existing participants TSV if one is found. "
            "Keep False when running in parallel to avoid race conditions."
        ),
    )
    parser.add_argument(
        "--add_sessions_tsv",
        dest="add_sessions_tsv",
        required=False,
        default=True,
        type=_convert_to_bool,
        help="Create a sessions TSV file containing the session and scan date for each subject.",
    )
    parser.add_argument(
        "--subjects_visits_file",
        dest="subjects_visits_file",
        required=True,
        type=str,
        help=(
            "Path to a CSV or Excel file mapping subjects to visit dates. "
            "Must contain 'participant_id' and 'date' columns. "
            "Include a 'dose' column to add dosages to the sessions TSV. "
            "For data from unwanted dates, set to a NULL value (leave that cell empty) or exclude that row from the data."
        ),
    )

    return parser


if __name__ == "__main__":
    args = _get_cmd_args().parse_args()
    run_pipeline(**vars(args))
