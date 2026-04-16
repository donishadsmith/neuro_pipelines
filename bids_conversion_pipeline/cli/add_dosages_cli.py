import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from add_dosages import run_pipeline


def _get_cmd_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add dosage information to sessions TSV files in a BIDS directory."
    )
    parser.add_argument(
        "--bids_dir",
        dest="bids_dir",
        required=True,
        help="The BIDS directory containing sessions TSV files.",
    )
    parser.add_argument(
        "--subjects",
        dest="subjects",
        required=False,
        nargs="+",
        default=None,
        help="One or more subject IDs (without the 'sub-' prefix) to restrict processing to.",
    )
    parser.add_argument(
        "--subjects_visits_file",
        dest="subjects_visits_file",
        required=True,
        help=(
            "Path to a CSV or Excel file mapping subjects to visit dates. "
            "Must contain 'participant_id', 'date', and 'dose' columns. "
            "Dates should be listed in chronological order per subject. "
            "Use NaN for missing sessions."
        ),
    )
    parser.add_argument(
        "--subjects_visits_date_fmt",
        dest="subjects_visits_date_fmt",
        required=False,
        default=r"%m/%d/%Y",
        help="Date format used in the subjects visits file (e.g., %%m/%%d/%%Y).",
    )
    parser.add_argument(
        "--sessions_tsv_date_fmt",
        dest="sessions_tsv_date_fmt",
        required=False,
        default=r"%y%m%d",
        help="Date format used in the sessions TSV files (e.g., %%y%%m%%d).",
    )

    return parser


if __name__ == "__main__":
    args = _get_cmd_args().parse_args()
    run_pipeline(**vars(args))
