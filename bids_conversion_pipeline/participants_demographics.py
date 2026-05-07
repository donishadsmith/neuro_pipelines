import re

import pandas as pd

from bidsaid.logging import setup_logger

LGR = setup_logger(__name__)


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


def _change_dtype(merged_df):
    for column in merged_df.columns:
        if pd.to_numeric(merged_df[column].dropna(), errors="coerce").notna().all():
            merged_df[column] = merged_df[column].astype(float)

    return merged_df


def run_pipeline(participants_tsv_path, demographics_file, covariates_to_add) -> None:
    participant_df = pd.read_csv(participants_tsv_path, sep="\t")
    demographic_df = _get_demographic_df(demographics_file)
    demographic_df.columns = [col.strip() for col in demographic_df.columns]
    if "participant_id" not in demographic_df.columns:
        raise ValueError("`participant_id` must be a column in `demographics_file`.")

    demographic_df["participant_id"] = (
        demographic_df["participant_id"]
        .astype(str)
        .apply(lambda x: re.findall(r"\d{5}", x)[0])
    )
    demographic_df = demographic_df.drop_duplicates(
        subset="participant_id", keep="first"
    )

    participant_df["participant_id"] = (
        participant_df["participant_id"]
        .astype(str)
        .apply(lambda x: re.findall(r"\d{5}", x)[0])
    )
    participant_df.columns = [col.lower() for col in participant_df.columns]

    covariates = [
        cov
        for cov in covariates_to_add
        if cov in demographic_df.columns and cov != "participant_id"
    ]
    missing_covariates = set(covariates_to_add).difference(covariates)
    if missing_covariates:
        LGR.info(
            f"The following covariates on not in `demographics_file`: {missing_covariates}"
        )

    if covariates:
        covariates += ["participant_id"]
        merged_df = pd.merge(
            participant_df, demographic_df[covariates], on="participant_id", how="left"
        )
        merged_df = merged_df.T.drop_duplicates().T
        merged_df.columns = [
            col.removesuffix("_x").removesuffix("_y")
            for col in merged_df.columns
            if col.removesuffix("_x").removesuffix("_y") in covariates
        ]
        merged_df["participant_id"] = merged_df["participant_id"].apply(
            lambda x: f"sub-{x}"
        )
        merged_df = _change_dtype(merged_df)
        merged_df.columns = [col.lower() for col in merged_df.columns]
        merged_df = merged_df.dropna(axis=1, how="all")

        LGR.info(f"Columns in 'participants.tsv': {merged_df.columns.to_list()}")

        merged_df.to_csv(participants_tsv_path, sep="\t", index=None)

        return merged_df.columns.tolist()
