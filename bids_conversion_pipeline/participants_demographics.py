import pandas as pd
from pandas.api.types import is_string_dtype

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


def run_pipeline(participants_tsv_path, demographics_file, covariates_to_add) -> None:
    participant_df = pd.read_csv(participants_tsv_path, sep="\t")
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
    participant_df.columns = [
        col.split(" ")[0] if col.startswith("age") else col
        for col in participant_df.columns
    ]

    participant_df.to_csv(participants_tsv_path, sep="\t", index=None)
