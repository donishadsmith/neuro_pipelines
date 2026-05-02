import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from subjects_visits import run_pipeline


def _get_cmd_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update or create the subjects visits file."
    )
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
        "--subjects_visits_file",
        dest="subjects_visits_file",
        required=False,
        type=str,
        help="Path to a CSV or Excel file mapping subjects to visit dates. ",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        required=False,
        default=None,
        type=str,
        help="Output directory for the subjects visits file.",
    )

    return parser


if __name__ == "__main__":
    args = _get_cmd_args().parse_args()
    run_pipeline(**vars(args))
