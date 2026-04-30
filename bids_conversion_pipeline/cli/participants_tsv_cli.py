import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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

    return parser


if __name__ == "__main__":
    args = _get_cmd_args().parse_args()
    run_pipeline(**vars(args))
