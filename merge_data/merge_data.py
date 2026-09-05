import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from _general_utils import (
    _get_dataframe,
    _standardize_dates,
    _standardize_participant_ids,
)

SEP_DICT = {"csv": ",", "tsv": "\t"}


def run_pipeline(
    primary_file,
    primary_file_subject_column,
    secondary_file,
    secondary_file_subject_column,
    columns_to_add,
    primary_file_date_column=None,
    secondary_file_date_column=None,
    primary_file_dose_column=None,
    secondary_file_dose_column=None,
    column_suffix=None,
    add_subject_prefix=True,
):
    primary_df = _get_dataframe(primary_file).drop_duplicates()
    primary_df = _standardize_participant_ids(primary_df, primary_file_subject_column)

    secondary_df = _get_dataframe(secondary_file).drop_duplicates()
    secondary_df = _standardize_participant_ids(
        secondary_df, secondary_file_subject_column
    )

    merge_columns = [primary_file_subject_column]
    rename_map = {secondary_file_subject_column: primary_file_subject_column}

    if primary_file_date_column and secondary_file_date_column:
        primary_df = _standardize_dates(
            primary_df,
            date_column_name=primary_file_date_column,
            participant_column_name=primary_file_subject_column,
        )
        secondary_df = _standardize_dates(
            secondary_df,
            date_column_name=secondary_file_date_column,
            participant_column_name=secondary_file_subject_column,
        )
        rename_map[secondary_file_date_column] = primary_file_date_column
        merge_columns.append(primary_file_date_column)

    if primary_file_dose_column and secondary_file_dose_column:
        rename_map[secondary_file_dose_column] = primary_file_dose_column
        merge_columns.append(primary_file_dose_column)

    secondary_df = secondary_df.rename(columns=rename_map)

    primary_df[primary_file_subject_column] = (
        primary_df[primary_file_subject_column].astype(int).astype(str)
    )
    secondary_df[primary_file_subject_column] = (
        secondary_df[primary_file_subject_column].astype(int).astype(str)
    )

    if column_suffix:
        suffix_map = {col: f"{col}{column_suffix}" for col in columns_to_add}
        secondary_df = secondary_df.rename(columns=suffix_map)
        columns_to_add = list(suffix_map.values())

    columns_to_add = [col for col in columns_to_add if col not in primary_df.columns]
    old_columns = primary_df.columns.tolist()

    merged_df = pd.merge(
        primary_df,
        secondary_df[columns_to_add + merge_columns].drop_duplicates(),
        on=merge_columns,
        how="left",
    ).drop_duplicates()

    new_columns = set(merged_df.columns.tolist()).difference(old_columns)
    if add_subject_prefix:
        merged_df[primary_file_subject_column] = "sub-" + merged_df[
            primary_file_subject_column
        ].astype(int).astype(str)

    merged_df.to_csv(
        primary_file,
        index=None,
        sep=SEP_DICT[Path(primary_file).suffix.removeprefix(".")],
        encoding="utf-8-sig",
    )

    return new_columns
