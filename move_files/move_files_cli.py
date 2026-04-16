import argparse

from move_files import run_pipeline

def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Move events or sessions files to BIDS directory."
    )
    parser.add_argument(
        "--src_dir",
        dest="src_dir",
        required=True,
        help=("Path containing the files. File naming should be BIDS compliant."),
    )
    parser.add_argument(
        "--bids_dir",
        dest="bids_dir",
        required=True,
        help="The root of the BIDS directory.",
    )

    return parser

if __name__ == "__main__":
    _get_cmd_args = _get_cmd_args()
    args = _get_cmd_args.parse_args()
    run_pipeline(**vars(args))
