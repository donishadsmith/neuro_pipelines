import argparse
from pathlib import Path

import pandas as pd
from pandas.api.types import is_string_dtype

from bidsaid.logging import setup_logger

from _utils import _create_or_append_participants_tsv

LGR = setup_logger(__name__)


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Create or append participants TSV in BIDS directory."
    )
    parser.add_argument(
        "--bids_dir", dest="bids_dir", required=True, help="The BIDS directory."
    )
    parser.add_argument(
        "--demographics_file",
        dest="demographics_file",
        required=False,
        default=None,
        help="A demographics file, subject IDs should be in a column names `participant_id`.",
    )
    parser.add_argument(
        "--covariates_to_add",
        dest="covariates_to_add",
        required=False,
        default=None,
        nargs="+",
        help="Names of the covariates from the demographics file.",
    )

    return parser


def _get_demographic_df(demographics_file):
    if str(demographics_file).endswith(".xlsx") or str(demographics_file).endswith(
        ".xls"
    ):
        return pd.read_excel(demographics_file)

    try:
        demographic_df = pd.read_csv(
            demographics_file, sep=None, engine="python", encoding="utf-8"
        )
    except UnicodeDecodeError:
        demographic_df = pd.read_csv(
            demographics_file, sep=None, engine="python", encoding="windows-1252"
        )

    return demographic_df


def _get_mask_and_change_dtype(participant_df, covariate):
    if covariate in participant_df.columns:
        mask = participant_df[covariate].isna()
        if (
            pd.to_numeric(participant_df[covariate].dropna(), errors="coerce")
            .notna()
            .all()
        ):
            participant_df[covariate] = participant_df[covariate].astype(float)
    else:
        mask = ~participant_df["participant_id"].isna()

    return mask, participant_df


def _check_new_categories(participant_df, covariate, covariate_values):
    if covariate not in participant_df.columns:
        return None

    if not is_string_dtype(participant_df[covariate]):
        return None

    unique_categories = participant_df[covariate].dropna().unique().tolist()
    if unique_categories:
        new_categories = [
            category
            for category in covariate_values
            if category not in unique_categories
        ]
        if new_categories:
            LGR.info(
                f"The following new categories will be added to {covariate}: {new_categories}"
            )


def main(bids_dir, demographics_file, covariates_to_add) -> None:
    if demographics_file and not covariates_to_add:
        raise ValueError(
            "`covariates_to_add` must be specified when `demographics_file` is used."
        )

    bids_dir = Path(bids_dir)
    participant_df = _create_or_append_participants_tsv(
        bids_dir,
        early_return=not bool(demographics_file),
        save_df=not bool(demographics_file),
        return_df=bool(demographics_file),
    )
    if participant_df is None:
        return None

    demographic_df = _get_demographic_df(demographics_file)
    if "participant_id" not in demographic_df.columns:
        raise ValueError("`participant_id` must be a column in `demographics_file`.")

    demographic_df["participant_id"] = demographic_df["participant_id"].astype(str)
    demographic_df = demographic_df.drop_duplicates(
        subset="participant_id", keep="first"
    )
    for covariate in covariates_to_add:
        if covariate not in demographic_df.columns:
            LGR.info(
                f"The following column name is not in `demographics_file`: {covariate}"
            )
            continue

        mask, participant_df = _get_mask_and_change_dtype(participant_df, covariate)

        participant_ids = participant_df.loc[mask, "participant_id"].tolist()
        if not all(demographic_df["participant_id"].isin(participant_ids).tolist()):
            participant_ids = [sub.removeprefix("sub-") for sub in participant_ids]

        covariate_values = demographic_df.loc[
            demographic_df["participant_id"].isin(participant_ids), covariate
        ].tolist()
        _check_new_categories(participant_df, covariate, covariate_values)

        participant_df.loc[mask, covariate] = covariate_values

    participant_df.columns = [col.lower() for col in participant_df.columns]
    participant_df.to_csv(bids_dir / "participants.tsv", sep="\t", index=None)


if __name__ == "__main__":
    args = _get_cmd_args().parse_args()
    main(**vars(args))
