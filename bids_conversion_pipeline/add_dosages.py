"""Standalone function when BIDS directory is created and sessions TSV file exists"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from bidsaid.files import get_entity_value
from bidsaid.logging import setup_logger

from _bids_conversion_utils import (
    _check_subjects_visits_file,
    _extract_subjects_visits_data,
    _standardize_dates,
    _strip_entity,
)

LGR = setup_logger(__name__)


def run_pipeline(
    bids_dir: str,
    subjects_visits_file: str | Path,
    subjects: list[str],
    subjects_visits_date_fmt: str,
    sessions_tsv_date_fmt: str,
) -> None:
    subjects_visits_df = _check_subjects_visits_file(
        subjects_visits_file, dose_column_required=True, return_df=True
    )
    subjects_visits_df["date"] = _standardize_dates(
        subjects_visits_df["date"], subjects_visits_date_fmt
    )

    sessions_tsv_list = Path(bids_dir).rglob("*sessions.tsv")
    if subjects:
        subjects = _strip_entity(subjects)
        sessions_tsv_list = [
            file
            for file in sessions_tsv_list
            if get_entity_value(file.name, "sub") in subjects
        ]

    # Converting date format from sessions tsv to subject tsv file, should technically be the same but include as a
    # parameter that can be changed
    change_date_fmt = lambda date: datetime.strptime(
        str(date), sessions_tsv_date_fmt
    ).strftime(subjects_visits_date_fmt)
    for sessions_tsv_file in sessions_tsv_list:
        sessions_tsv_df = pd.read_csv(sessions_tsv_file, sep=None, engine="python")
        if "acq_time" not in sessions_tsv_df.columns:
            LGR.critical(
                f"Skipping {sessions_tsv_file}. The sessions TSV file must have an 'acq_time' column."
            )
            continue

        if "dose" not in sessions_tsv_df.columns:
            sessions_tsv_df["dose"] = float("NaN")

        sessions_tsv_df["dose"] = sessions_tsv_df["dose"].astype(str)
        cleaned_sessions_tsv_df = sessions_tsv_df.dropna(subset="acq_time")
        original_dates = [
            str(val) for val in cleaned_sessions_tsv_df["acq_time"].to_numpy(copy=True)
        ]
        for original_date, converted_date in zip(
            original_dates, map(change_date_fmt, original_dates)
        ):
            dose = _extract_subjects_visits_data(
                participant_id=get_entity_value(sessions_tsv_file.name, "sub"),
                subjects_visits_df=subjects_visits_df,
                column_name="dose",
                subjects_visits_date_fmt=subjects_visits_date_fmt,
                scan_date=converted_date,
            )
            if dose:
                dose = dose[0]
            else:
                continue

            mask = sessions_tsv_df["acq_time"].astype(str) == str(original_date)
            row_id = mask.tolist().index(True)
            sessions_tsv_df.at[row_id, "dose"] = dose

        sessions_tsv_df = add_dose_mg(
            sessions_tsv_file, sessions_tsv_df, subjects_visits_df
        )

        sessions_tsv_df.to_csv(sessions_tsv_file, sep="\t", index=False)


def add_dose_mg(sessions_tsv_file, sessions_tsv_df, subjects_visits_df):
    if not all(
        col in subjects_visits_df.columns for col in ["participant_id", "dose_mg"]
    ):
        return sessions_tsv_df

    subjects_visits_df["participant_id"] = subjects_visits_df["participant_id"].astype(
        str
    )
    subjects_visits_df["dose_mg"] = subjects_visits_df["dose_mg"].astype(int)

    subject = get_entity_value(sessions_tsv_file, "sub")
    sessions_tsv_df.loc[sessions_tsv_df["dose"] == "placebo", "dose_mg"] = 0
    non_placebo_dose = subjects_visits_df.loc[
        (subjects_visits_df["participant_id"] == subject)
        & (subjects_visits_df["dose_mg"] != 0),
        "dose_mg",
    ].values
    sessions_tsv_df.loc[sessions_tsv_df["dose"] == "mph", "dose_mg"] = non_placebo_dose

    return sessions_tsv_df
