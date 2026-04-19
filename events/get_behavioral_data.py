"""
Since ``create_event_files.py`` already exists, a lazy implementation of getting accuracy and reaction times will be executed.

Essentially, this pipeline runs the pipeline in ``create_event_files.py``, then performs I/O operations to load in the events TSV files
and append data to a CSV file.
"""

import numpy as np, pandas as pd

from bidsaid.files import get_entity_value
from bidsaid.logging import setup_logger

from create_event_files import run_pipeline as run_events_pipeline

BLOCK_TASKS = {
    "kids": ["mtle", "mtlr", "nback", "princess"],
    "adults": ["mtle", "mtlr", "nback"],
}

LGR = setup_logger(__name__)


def run_pipeline(
    log_dir,
    dst_dir,
    temp_dir,
    delete_temp_dir,
    task,
    cohort,
    subjects,
    minimum_file_size,
    subjects_visits_file,
    subjects_visits_date_fmt,
    behavioral_data_file,
    exclude_filenames,
    caller,
):

    dst_dir = run_events_pipeline(
        log_dir,
        dst_dir,
        temp_dir,
        delete_temp_dir,
        task,
        cohort,
        subjects,
        minimum_file_size,
        subjects_visits_file,
        subjects_visits_date_fmt,
        exclude_filenames,
        caller=caller,
    )

    is_event_task = task not in BLOCK_TASKS[cohort]
    behavioral_data_dict = {"participant_id": [], "session_id": []}
    if behavioral_data_file:
        behavioral_df = pd.read_csv(behavioral_data_file, sep=None, engine="python")

    events_files = list(dst_dir.glob(f"*{task}*_events.tsv"))

    LGR.info("Creating behavioral data...")

    for index, event_file in enumerate(events_files):
        participant_id = get_entity_value(event_file, "sub")
        session_id = get_entity_value(event_file, "ses")
        if behavioral_data_file:
            is_empty = behavioral_df[
                (behavioral_df["participant_id"] == participant_id)
                & (behavioral_df["session_id"] == session_id)
            ].empty
            if not is_empty:
                continue

        behavioral_data_dict["participant_id"].append(participant_id)
        behavioral_data_dict["session_id"].append(session_id)

        events_df = pd.read_csv(event_file, sep="\t")
        trial_types = [
            x for x in events_df["trial_type"].unique() if "instruction" not in x
        ]

        if index == 0:
            column_types = [
                "accuracy",
                "overall_reaction_time_mean",
                "overall_reaction_time_std",
            ]
            if is_event_task:
                column_types += [
                    "correct_reaction_time_mean",
                    "correct_reaction_time_std",
                    "error_reaction_time_mean",
                    "error_reaction_time_std",
                ]

            behavioral_data_dict.update(
                {
                    f"{trial_type}_{column_type}": []
                    for trial_type in trial_types
                    for column_type in column_types
                }
            )

        for trial_type in trial_types:
            condition_df = events_df[events_df["trial_type"] == trial_type].copy()
            if is_event_task:
                condition_df["accuracy"] = condition_df["accuracy"].replace(
                    {"correct": 1, "incorrect": 0}
                )

            behavioral_data_dict[f"{trial_type}_accuracy"].append(
                np.nanmean(condition_df["accuracy"])
            )

            if not is_event_task and "response_count" in condition_df.columns:
                weights = condition_df["response_count"]
                rts = condition_df["reaction_time"]
                valid = weights.notna() & rts.notna() & (weights > 0)
                if valid.any():
                    weighted_rt_mean = (weights[valid] * rts[valid]).sum() / weights[
                        valid
                    ].sum()
                else:
                    weighted_rt_mean = np.nan

                behavioral_data_dict[f"{trial_type}_overall_reaction_time_mean"].append(
                    weighted_rt_mean
                )
            else:
                behavioral_data_dict[f"{trial_type}_overall_reaction_time_mean"].append(
                    np.nanmean(condition_df["reaction_time"])
                )

            behavioral_data_dict[f"{trial_type}_overall_reaction_time_std"].append(
                np.nanstd(condition_df["reaction_time"])
            )
            if is_event_task:
                behavioral_data_dict[f"{trial_type}_correct_reaction_time_mean"].append(
                    np.nanmean(
                        condition_df.loc[condition_df["accuracy"] == 1, "reaction_time"]
                    )
                )
                behavioral_data_dict[f"{trial_type}_correct_reaction_time_std"].append(
                    np.nanstd(
                        condition_df.loc[condition_df["accuracy"] == 1, "reaction_time"]
                    )
                )
                behavioral_data_dict[f"{trial_type}_error_reaction_time_mean"].append(
                    np.nanmean(
                        condition_df.loc[condition_df["accuracy"] == 0, "reaction_time"]
                    )
                )
                behavioral_data_dict[f"{trial_type}_error_reaction_time_std"].append(
                    np.nanstd(
                        condition_df.loc[condition_df["accuracy"] == 0, "reaction_time"]
                    )
                )

        event_file.unlink()

    df = pd.DataFrame(behavioral_data_dict)
    if behavioral_data_file:
        df = pd.concat([behavioral_df, df], axis=0, ignore_index=True)
        df.to_csv(behavioral_data_file, sep=",", index=None)
    else:
        dst_dir = dst_dir / "CSV"
        dst_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            dst_dir / f"cohort-{cohort}_task-{task}_desc-behavioral_data.csv",
            sep=",",
            index=None,
        )

    return dst_dir
