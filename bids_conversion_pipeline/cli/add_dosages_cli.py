import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
            "Must contain 'participant_id' and 'date' columns. "
            "Include a 'dose' column to add dosages to the sessions TSV. "
            "For data from unwanted dates, set to a NULL value (leave that cell empty) or exclude that row from the data."
            "If `dose_mg` (only relevant to adult cohort since the dose column is coded as 'mph' and 'placebo') "
            "is a column in the file, then that information will be included too."
        ),
    )

    return parser


if __name__ == "__main__":
    args = _get_cmd_args().parse_args()
    run_pipeline(**vars(args))
