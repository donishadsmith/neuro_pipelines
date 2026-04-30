from typing import Optional

import pandas as pd
from bidsaid.path_utils import is_valid_date
from bidsaid.logging import setup_logger

LGR = setup_logger(__name__)


def _extract_subjects_visits_data(
    participant_id: str,
    subjects_visits_df: pd.DataFrame,
    column_name: int,
    scan_date: Optional[str] = None,
    date_column_name: str = "date",
):
    mask = subjects_visits_df["participant_id"] == str(participant_id)
    if scan_date:
        mask &= subjects_visits_df[date_column_name] == str(scan_date)

    subjects_visits_df[mask].loc[:, column_name].astype(str).tolist()

    return subjects_visits_df[mask].loc[:, column_name].astype(str).tolist()


def _get_subject_visits(
    participant_id: str,
    subjects_visits_df: pd.DataFrame,
) -> dict[str, str]:
    visit_dates = _extract_subjects_visits_data(
        participant_id,
        subjects_visits_df,
        column_name="date",
    )
    if not visit_dates:
        LGR.warning(f"Subject {participant_id} has no visit dates.")

        return None

    return {
        f"0{session_id}": date for session_id, date in enumerate(visit_dates, start=1)
    }


def _standardize_dates(
    df: pd.DataFrame, date_column_name="date", sort_data=True
) -> list[str]:
    df.columns = [col.strip() for col in df.columns]

    if "participant_id" in df.columns:
        df[["participant_id", date_column_name]] = df[
            ["participant_id", date_column_name]
        ].astype(str)
    else:
        df[date_column_name] = df[date_column_name].astype(str)

    null_dates = df[date_column_name].isna()
    if any(null_dates.tolist()):
        LGR.info("Dropping Null/NaN dates in subjects visits data")

    df = df[~null_dates]
    df[date_column_name] = [
        (
            str(pd.to_datetime([date_str], format=r"%y%m%d")[0]).split()[0]
            if is_valid_date(date_str, r"%y%m%d")
            else str(pd.to_datetime([date_str], errors="coerce")[0]).split()[0]
        )
        for date_str in df[date_column_name]
    ]

    invalid_dates = df[date_column_name] == "NaT"
    if any(invalid_dates.tolist()):
        LGR.info(
            f"Dropping the following invalid dates in subjects visits data: {df.loc[invalid_dates, date_column_name]}"
        )

    df = df[~invalid_dates]

    if sort_data:
        if "participant_id" in df.columns:
            df = df.sort_values(by=["participant_id", date_column_name], ascending=True)
        else:
            df = df.sort_values(by=[date_column_name], ascending=True)

    return df
