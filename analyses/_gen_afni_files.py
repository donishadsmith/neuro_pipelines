import subprocess
from pathlib import Path

import numpy as np, pandas as pd

from bidsaid.logging import setup_logger

from _denoising import get_col_name, get_new_matrix_and_names, remove_collinear_columns
from _utils import CONDITION_DURATIONS

LGR = setup_logger(__name__)


def create_censor_file(subject_dir, subject, session, task, space, censor_mask):
    censor_file = (
        subject_dir
        / f"sub-{subject}_ses-{session}_task-{task}_run-01_space-{space}_desc-censor.1D"
    )
    np.savetxt(censor_file, censor_mask, fmt="%d")

    return censor_file


def create_nuisance_regressor_file(
    subject_dir,
    subject,
    session,
    task,
    space,
    censor_mask,
    regressor_names,
    *regressor_arrays,
    regressor_file_prefix="3ddeconvolve",
    analysis_type="glm",
):
    regressor_positions = {pos: name for pos, name in enumerate(regressor_names)}
    dependent_var = "BOLD/NIfTI image"
    if analysis_type == "gPPI":
        dependent_var += " and seed timeseries"

    LGR.info(
        f"Regressor names for the {dependent_var} and positions: {regressor_positions}"
    )
    regressor_file = (
        subject_dir
        / f"sub-{subject}_ses-{session}_task-{task}_run-01_space-{space}_desc-{regressor_file_prefix}_nuisance_regressors.1D"
    )
    valid_arrays = [arr for arr in regressor_arrays if arr is not None]
    data = np.column_stack(valid_arrays)

    rank = np.linalg.matrix_rank(data)
    collinear_regressor_names = []
    if rank < data.shape[1]:
        LGR.warning(f"Regressor matrix is rank deficient: {rank}")

        data, regressor_positions, collinear_regressor_names = remove_collinear_columns(
            data, regressor_positions
        )
        LGR.warning(f"New regressor names and positions: {regressor_positions}")

    # To stop errors and warnings
    drop_columns = np.where(np.var(data, axis=0) == 0)[0].tolist()
    if drop_columns:
        constant_col_names = [
            get_col_name(indx, regressor_positions) for indx in drop_columns
        ]

        LGR.warning(f"The following have constant variance: {constant_col_names}")

        data, regressor_positions = get_new_matrix_and_names(
            drop_columns, data, regressor_positions
        )

        LGR.warning(f"New regressor names and positions: {regressor_positions}")
    else:
        constant_col_names = []

    censored_volumes = np.where(censor_mask == 0)[0].tolist()
    LGR.info(f"The following volume indices will be censored: {censored_volumes}")

    mean = data[censor_mask.astype(bool)].mean(axis=0)
    std = data[censor_mask.astype(bool)].std(axis=0, ddof=1)
    std[std < np.finfo(np.float64).eps] = 1.0
    data[censor_mask.astype(bool)] = (data[censor_mask.astype(bool)] - mean) / std

    np.savetxt(regressor_file, data, fmt="%.6f")

    report_info = {
        "collinear_regressor_names": collinear_regressor_names,
        "constant_column_names": constant_col_names,
    }

    return regressor_file, report_info


def is_timing_file_empty(timing_file):
    if not Path(timing_file).exists():
        return True

    return np.loadtxt(timing_file, delimiter=" ").size == 0


def save_event_file(timing_dir, trial_type, timing_data):
    filename = timing_dir / f"{trial_type}.1D"
    timing_str = " ".join(timing_data)
    with open(filename, "w") as f:
        f.write(f"{timing_str}")


def create_binary_condition(
    afni_img_path, timing_dir, cohort, task, tr, n_volumes, censor_file
):

    censor_vector = np.loadtxt(censor_file, dtype=int)

    condition_filenames_dict = {}
    for timing_file in timing_dir.glob("*.1D"):
        condition_name = timing_file.name.removesuffix(".1D")

        noncensored_condition_filename = (
            timing_dir
            / "censored"
            / f"{condition_name}_desc-noncensored_binary_vector.1D"
        )
        censored_condition_filename = (
            noncensored_condition_filename.parent
            / f"{condition_name}_desc-censored_binary_vector.1D"
        )

        condition_filenames_dict.update(
            {
                condition_name: {
                    "noncensored_binary_vector": noncensored_condition_filename,
                    "censored_binary_vector": censored_condition_filename,
                }
            }
        )

        censored_condition_filename.mkdir(parents=True, exist_ok=True)

        duration = (
            CONDITION_DURATIONS[cohort][task]
            if not condition_name.startswith("instruction")
            else CONDITION_DURATIONS[cohort][f"{condition_name}_{task}"]
        )

        cmd = (
            f"apptainer exec -B /projects:/projects {afni_img_path} timing_tool.py "
            f"-timing {timing_file} "
            f"-tr {tr} "
            f"-stim_dur {duration} "
            f"-run_len {tr * n_volumes} "
            f"-timing_to_1D {noncensored_condition_filename}"
        )

        LGR.info(
            f"Creating binary vector for diagnostic plotting: {condition_name}: {cmd}"
        )
        subprocess.run(cmd, shell=True, check=True)

        # Now add censoring
        condition_vector = np.loadtxt(noncensored_condition_filename)
        condition_vector[censor_vector == 0] = 0

        np.savetxt(
            censored_condition_filename, condition_vector.reshape(-1, 1), fmt="%f"
        )

    return condition_filenames_dict


def create_timing_files(
    subject_dir, event_file, task, filter_correct_trials=False, append_task_name=True
):
    event_related_tasks = ["flanker", "simplegng", "complexgng"]
    special_tasks = event_related_tasks if filter_correct_trials else []
    if task in event_related_tasks:
        if filter_correct_trials:
            LGR.info(
                f"**FILTERING** the following task for correct trials only: {task}. "
                "Contrasts related to this task will **ONLY** include trials that subject "
                "answered correctly on."
            )
        else:
            LGR.info(
                f"**NOT FILTERING** the following task for correct trials only: {task} "
                "Contrasts related to this task will include **ALL** trials for subject "
                "regardless if they answered correctly or incorrectly."
            )

    timing_dir = subject_dir / "timing_files" / (task if append_task_name else "")
    timing_dir.mkdir(parents=True, exist_ok=True)

    event_df = pd.read_csv(event_file, sep="\t")
    trial_types = event_df["trial_type"].unique()

    for trial_type in trial_types:
        trial_df = event_df[event_df["trial_type"] == trial_type]
        row_mask = (
            np.full(len(trial_df), True, dtype=bool)
            if task not in special_tasks
            else trial_df["accuracy"] == "correct"
        )

        timing_data = (
            trial_df.loc[
                row_mask,
                "onset",
            ]
            .astype(str)
            .to_numpy(copy=True)
        )

        if isinstance(timing_data, pd.Series):
            timing_data = timing_data.to_list()

        save_event_file(timing_dir, trial_type, timing_data)

    # Get all errors
    if task in special_tasks:
        timing_data = event_df.loc[event_df["accuracy"] == "incorrect", "onset"].astype(
            str
        )

        save_event_file(timing_dir, trial_type="errors", timing_data=timing_data)

    return timing_dir
