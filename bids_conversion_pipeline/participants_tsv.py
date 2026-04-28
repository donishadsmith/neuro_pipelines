from pathlib import Path

from _bids_conversion_utils import _create_or_append_participants_tsv


def run_pipeline(bids_dir):
    bids_dir = Path(bids_dir)
    _create_or_append_participants_tsv(bids_dir)
