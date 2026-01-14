"""Standalone function when BIDS directory is created and sessions TSV file exists"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from nifti2bids.bids import get_entity_value
from nifti2bids.logging import setup_logger

from _utils import (
    _check_subjects_visits_file,
    _extract_subjects_visits_data,
    _standardize_dates,
    _strip_entity,
)

LGR = setup_logger(__name__)


def _get_cmd_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add dosage information to a sessions TSV file."
    )
    parser.add_argument(
        "--bids_dir", dest="bids_dir", required=True, help="The BIDS directory."
    )
    parser.add_argument(
        "--subjects",
        dest="subjects",
        required=False,
        nargs="+",
        default=None,
        help="The subject IDs in the 'src_dir' to convert to BIDS.",
    )
    parser.add_argument(
        "--subjects_visits_file",
        dest="subjects_visits_file",
        required=True,
        help=(
            "A text file, where the 'subject_id' contaims the subject ID and the "
            "'date' column is the date of visit. Using this parameter is recommended "
            "when data is missing. Ensure all dates have a consistent format. "
            "**All subject visit dates should be listed.** If a 'dose' column is included, "
            "then dosages will be included in the sessions TSV file."
        ),
    )
    parser.add_argument(
        "--subjects_visits_date_fmt",
        dest="subjects_visits_date_fmt",
        required=False,
        default=r"%m/%d/%Y",
        help=("The format of the date in ``subjects_visits_file``."),
    )
    parser.add_argument(
        "--sessions_tsv_date_fmt",
        dest="sessions_tsv_date_fmt",
        required=False,
        default=r"%y%m%d",
        help=("The format of the date in the sessions TSV files."),
    )

    return parser


def add_dosages_to_sessions_tsv(
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
            str(val) for val in cleaned_sessions_tsv_df["acq_time"].values
        ]
        for original_date, converted_date in zip(
            original_dates, map(change_date_fmt, original_dates)
        ):
            dose = _extract_subjects_visits_data(
                subject_id=get_entity_value(sessions_tsv_file.name, "sub"),
                subjects_visits_df=subjects_visits_df,
                column_name="dose",
                scan_date=converted_date,
            )
            if dose:
                dose = dose[0]
            else:
                continue

            mask = sessions_tsv_df["acq_time"].astype(str) == str(original_date)
            row_id = mask.tolist().index(True)
            sessions_tsv_df.at[row_id, "dose"] = dose

        sessions_tsv_df.to_csv(sessions_tsv_file, sep="\t", index=False)


if __name__ == "__main__":
    args = _get_cmd_args().parse_args()
    add_dosages_to_sessions_tsv(**vars(args))
