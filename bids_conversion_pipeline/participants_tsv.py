import argparse

from pathlib import Path

from _utils import _create_or_append_participants_tsv

parser = argparse.ArgumentParser(description="Create or append participants TSV in BIDS directory.")
parser.add_argument(
    "--bids_dir", dest="bids_dir", required=True, help="The BIDS directory."
)

_create_or_append_participants_tsv(Path(parser.parse_args().bids_dir))