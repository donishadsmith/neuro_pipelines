from pathlib import Path

import pandas as pd

from bidsaid.files import create_participant_tsv
from bidsaid.logging import setup_logger

LGR = setup_logger(__name__)


def _create_or_append_participants_tsv(
    bids_dir: Path,
) -> None:
    if not (tsv_file := list(bids_dir.glob("participants.tsv"))):
        participants_df = create_participant_tsv(
            bids_dir, save_df=False, return_df=True
        )
    else:
        participants_df = pd.read_csv(tsv_file[0], sep="\t")

    new_participants_df = create_participant_tsv(
        bids_dir, save_df=False, return_df=True
    )
    missing_participants = list(
        set(new_participants_df["participant_id"]).difference(
            participants_df["participant_id"]
        )
    )
    missing_participants_mask = new_participants_df["participant_id"].isin(
        missing_participants
    )
    new_participants_df = new_participants_df[missing_participants_mask]

    if not new_participants_df.empty:
        combined_participants_df = pd.concat(
            [participants_df, new_participants_df], ignore_index=True
        )
    else:
        combined_participants_df = participants_df

    combined_participants_df.to_csv(bids_dir / "participants.tsv", sep="\t", index=None)


def _strip_entity(subjects: list[str | int]) -> list[str]:
    return [str(subject).removeprefix("sub-") for subject in subjects]
