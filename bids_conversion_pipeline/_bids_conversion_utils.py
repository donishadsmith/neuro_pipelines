import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from bidsaid.files import create_participant_tsv
from bidsaid.logging import setup_logger

from _exceptions import SubjectsVisitsFileError

LGR = setup_logger(__name__)


def _create_or_append_participants_tsv(
    bids_dir: Path,
    early_return: bool = True,
    save_df: bool = True,
    return_df: bool = False,
) -> None | pd.DataFrame:
    if not (tsv_file := list(bids_dir.glob("participants.tsv"))):
        participants_df = create_participant_tsv(
            bids_dir, save_df=False, return_df=True
        )
        if early_return:
            return None
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

    if save_df:
        combined_participants_df.to_csv(
            bids_dir / "participants.tsv", sep="\t", index=None
        )

    return combined_participants_df if return_df else None


def _standardize_dates(dates: list[str | int | float], fmt: str) -> list[str | float]:
    dates = [str(x).strip().split(" ")[0] for x in dates]
    convert_date = lambda date: (
        datetime.strptime(str(date), fmt).strftime(fmt)
        if not str(date).lower() == "nan"  # Check for the NaN case
        else float("NaN")
    )

    return list(map(convert_date, dates))


def _extract_subjects_visits_data(
    participant_id: str,
    subjects_visits_df: pd.DataFrame,
    column_name: int,
    subjects_visits_date_fmt: Optional[str] = None,
    scan_date: Optional[str] = None,
):
    subjects_visits_df.columns = [col.strip() for col in subjects_visits_df.columns]
    subjects_visits_df[["participant_id", "date"]] = subjects_visits_df[
        ["participant_id", "date"]
    ].astype(str)

    if (
        subjects_visits_date_fmt
        and bool(re.search(r"\s", subjects_visits_date_fmt)) is False
    ):
        subjects_visits_df["date"] = subjects_visits_df["date"].str.replace(
            r"\s+", "", regex=True
        )

    mask = subjects_visits_df["participant_id"] == str(participant_id)

    if scan_date:
        mask &= subjects_visits_df["date"] == str(scan_date)

    return (
        subjects_visits_df[mask]
        .loc[:, column_name]
        .astype(str)
        .to_numpy(copy=True)
        .tolist()
    )


def _strip_entity(subjects: list[str | int]) -> list[str]:
    return [str(subject).removeprefix("sub-") for subject in subjects]


def _check_subjects_visits_file(
    subjects_visits_file: str | Path,
    dose_column_required: bool,
    return_df: bool = False,
) -> None | pd.DataFrame:
    required_colnames = ["participant_id", "date"]

    if str(subjects_visits_file).endswith(".xlsx") or str(
        subjects_visits_file
    ).endswith(".xls"):
        subjects_visits_df = pd.read_excel(subjects_visits_file)
    else:
        subjects_visits_df = pd.read_csv(
            subjects_visits_file, sep=None, engine="python"
        )

    subjects_visits_df.columns = [col.strip() for col in subjects_visits_df.columns]

    if not all(
        required_colname in subjects_visits_df.columns.tolist()
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

    subjects_visits_df["participant_id"] = subjects_visits_df["participant_id"].astype(
        str
    )

    return subjects_visits_df if return_df else None
