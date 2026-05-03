"""Standalone function when BIDS directory is created and sessions TSV file exists"""

import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from bidsaid.files import get_entity_value
from bidsaid.logging import setup_logger

from _bids_conversion_utils import (
    _strip_entity,
)

from _general_utils import (
    _check_subjects_visits_file,
    _extract_subjects_visits_data,
    _standardize_dates,
)

LGR = setup_logger(__name__)


def run_pipeline(
    bids_dir: str,
    subjects_visits_file: str | Path,
    subjects: list[str],
) -> None:
    subjects_visits_df = _check_subjects_visits_file(
        subjects_visits_file, dose_column_required=True, return_df=True
    )
    subjects_visits_df = _standardize_dates(subjects_visits_df)

    sessions_tsv_list = Path(bids_dir).rglob("*sessions.tsv")
    if subjects:
        subjects = _strip_entity(subjects)
        subjects = [re.findall(r"\d{5}", x)[0] for x in subjects]
        sessions_tsv_list = [
            file
            for file in sessions_tsv_list
            if get_entity_value(file.name, "sub") in subjects
        ]

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
        cleaned_sessions_tsv_df = _standardize_dates(
            cleaned_sessions_tsv_df, date_column_name="acq_time", sort_data=False
        )
        for date in cleaned_sessions_tsv_df["acq_time"].tolist():
            dose = _extract_subjects_visits_data(
                participant_id=get_entity_value(sessions_tsv_file.name, "sub"),
                subjects_visits_df=subjects_visits_df,
                column_name="dose",
                scan_date=date,
            )
            if not dose:
                continue

            mask = cleaned_sessions_tsv_df["acq_time"] == date
            row_id = mask.tolist().index(True)
            cleaned_sessions_tsv_df.at[row_id, "dose"] = dose[0]

        cleaned_sessions_tsv_df = add_dose_mg(
            sessions_tsv_file, cleaned_sessions_tsv_df, subjects_visits_df
        )

        cleaned_sessions_tsv_df.to_csv(sessions_tsv_file, sep="\t", index=False)


def add_dose_mg(sessions_tsv_file, cleaned_sessions_tsv_df, subjects_visits_df):
    if not all(
        col in subjects_visits_df.columns for col in ["participant_id", "dose_mg"]
    ):
        return cleaned_sessions_tsv_df

    subjects_visits_df["participant_id"] = subjects_visits_df["participant_id"].astype(
        str
    )
    subjects_visits_df["dose_mg"] = subjects_visits_df["dose_mg"].astype(int)

    subject = get_entity_value(sessions_tsv_file, "sub")
    cleaned_sessions_tsv_df.loc[
        cleaned_sessions_tsv_df["dose"] == "placebo", "dose_mg"
    ] = 0
    non_placebo_dose = subjects_visits_df.loc[
        (subjects_visits_df["participant_id"] == subject)
        & (subjects_visits_df["dose_mg"] != 0),
        "dose_mg",
    ].values

    cleaned_sessions_tsv_df.loc[cleaned_sessions_tsv_df["dose"] == "mph", "dose_mg"] = (
        non_placebo_dose
    )

    return cleaned_sessions_tsv_df
