import argparse
from pathlib import Path

import pandas as pd

from nifti2bids.bids import get_entity_value


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Add dose mg to sessions TSV of adult data."
    )
    parser.add_argument(
        "--bids_dir", dest="bids_dir", required=True, help="The BIDS directory."
    )
    parser.add_argument(
        "--dose_file",
        dest="dose_file",
        required=True,
        help=(
            "A dose file, subject IDs should be in a column names `participant_id` "
            "and and dose mg in a column named `dose_mg`."
        ),
    )

    return parser


def _get_dose_df(dose_file):
    if str(dose_file).endswith(".xlsx") or str(dose_file).endswith(".xls"):
        return pd.read_excel(dose_file)

    try:
        dose_df = pd.read_csv(dose_file, sep=None, engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        dose_df = pd.read_csv(
            dose_file, sep=None, engine="python", encoding="windows-1252"
        )

    return dose_df


def main(bids_dir, dose_file):
    bids_dir = Path(bids_dir)

    dose_df = _get_dose_df(dose_file)

    required_cols = ["participant_id", "dose_mg"]
    if not all(col in dose_df.columns for col in ["participant_id", "dose_mg"]):
        raise ValueError(
            f"`dose_file` must have the following columns: {required_cols}"
        )

    dose_df["participant_id"] = dose_df["participant_id"].astype(str)

    for session_tsv in bids_dir.rglob("*sessions*.tsv"):
        subject = get_entity_value(session_tsv, "sub")
        sessions_df = pd.read_csv(session_tsv, sep=None, engine="python")
        sessions_df.loc[sessions_df["dose"] == "placebo", "dose_mg"] = 0
        non_placebo_dose = dose_df.loc[
            (dose_df["participant_id"] == subject) & (dose_df["dose_mg"] != 0),
            "dose_mg",
        ].values
        sessions_df.loc[sessions_df["dose"] == "mph", "dose_mg"] = non_placebo_dose
        sessions_df.to_csv(session_tsv, sep="\t", index=None)


if __name__ == "__main__":
    args = _get_cmd_args().parse_args()
    main(**vars(args))
