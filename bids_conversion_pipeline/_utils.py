from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from nifti2bids.bids import create_participant_tsv
from nifti2bids.logging import setup_logger

from _exceptions import SubjectsVisitsFileError

LGR = setup_logger(__name__)


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


def _standardize_dates(dates: list[str | int | float], fmt: str) -> list[str | float]:
    convert_date = lambda date: (
        datetime.strptime(str(date), fmt).strftime(fmt)
        if not str(date).lower() == "nan"  # Check for the NaN case
        else float("NaN")
    )

    return list(map(convert_date, dates))


def _extract_subjects_visits_data(
    subject_id: str,
    subjects_visits_df: pd.DataFrame,
    column_name: int,
    scan_date: Optional[str] = None,
):
    mask = subjects_visits_df["subject_id"].astype(str) == subject_id

    if scan_date:
        mask &= subjects_visits_df["date"].astype(str) == str(scan_date)

    return subjects_visits_df[mask].loc[:, column_name].astype(str).to_numpy(copy=True).tolist()


def _strip_entity(subjects: list[str | int]) -> list[str]:
    return [str(subject).removeprefix("sub-") for subject in subjects]


def _check_subjects_visits_file(
    subjects_visits_file: str | Path,
    dose_column_required: bool,
    return_df: bool = False,
) -> None | pd.DataFrame:
    required_colnames = ["subject_id", "date"]

    subjects_visits_df = pd.read_csv(subjects_visits_file, sep=None, engine="python")
    if not all(
        required_colname in subjects_visits_df.columns
        for required_colname in required_colnames
    ):
        raise SubjectsVisitsFileError(
            f"The following columns are required in {subjects_visits_file}: "
            f"{required_colnames}."
        )

    if dose_column_required:
        if "dose" not in subjects_visits_df.columns:
            raise SubjectsVisitsFileError(
                f"A 'dose' column is required in {subjects_visits_file}"
            )

    if len(subjects_visits_df.columns) > 2 and "dose" not in subjects_visits_df.columns:
        LGR.warning(
            f"More than two columns detected in {subjects_visits_file} but 'dose' "
            "column is missing. To include 'dose' values in the session TSV files, this column "
            "must be included."
        )

    return subjects_visits_df if return_df else None
