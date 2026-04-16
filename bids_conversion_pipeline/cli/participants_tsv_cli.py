import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from participants_tsv import run_pipeline


def _get_cmd_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or update the participants TSV file in a BIDS directory."
    )
    parser.add_argument(
        "--bids_dir",
        dest="bids_dir",
        required=True,
        help="The BIDS directory.",
    )
    parser.add_argument(
        "--demographics_file",
        dest="demographics_file",
        required=False,
        default=None,
        help=(
            "Path to a demographics file. " "Must contain a 'participant_id' column."
        ),
    )
    parser.add_argument(
        "--covariates_to_add",
        dest="covariates_to_add",
        required=False,
        default=None,
        nargs="+",
        help="Column names from the demographics file to add to the participants TSV.",
    )

    return parser


if __name__ == "__main__":
    args = _get_cmd_args().parse_args()
    run_pipeline(**vars(args))
