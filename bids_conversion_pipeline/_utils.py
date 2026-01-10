from pathlib import Path
from typing import Literal

import pandas as pd

from nifti2bids.bids import create_participant_tsv

def _get_constant(
    object: dict[str, list[str]] | dict[str, int | float],
    dataset: Literal["mph", "naag"],
    cohort: Literal["kids", "adults"],
) -> list[str] | dict[str, int]:
    constant = object[dataset]
    if dataset == "mph":
        constant = constant[cohort]

    return constant

def _create_or_append_participants_tsv(bids_dir: Path) -> None:
    if not (tsv_file := list(bids_dir.glob("participants.tsv"))):
        create_participant_tsv(bids_dir, save_df=True, return_df=False)
    else:
        new_participants_df = create_participant_tsv(
            bids_dir, save_df=False, return_df=True
        )
        old_participants_df = pd.read_csv(tsv_file[0], sep="\t")

        missing_participants = list(
            set(new_participants_df["participant_id"]).difference(
                old_participants_df["participant_id"]
            )
        )
        missing_participants_mask = new_participants_df["participant_id"].isin(
            missing_participants
        )
        new_participants_df = new_participants_df[missing_participants_mask]

        if not new_participants_df.empty:
            combined_participants_df = pd.concat(
                [old_participants_df, new_participants_df], ignore_index=True
            )
            combined_participants_df.to_csv(bids_dir / "participants.tsv", sep="\t")

