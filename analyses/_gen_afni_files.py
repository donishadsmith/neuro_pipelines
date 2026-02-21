import numpy as np, pandas as pd

from nifti2bids.logging import setup_logger

from _denoising import get_col_name, get_new_matrix_and_names, remove_collinear_columns

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
):
    regressor_positions = {pos: name for pos, name in enumerate(regressor_names)}
    LGR.info(f"Regressor names and positions: {regressor_positions}")
    regressor_file = (
        subject_dir
        / f"sub-{subject}_ses-{session}_task-{task}_run-01_space-{space}_desc-{regressor_file_prefix}_nuisance_regressors.1D"
    )
    valid_arrays = [arr for arr in regressor_arrays if arr is not None]
    data = np.column_stack(valid_arrays)

    rank = np.linalg.matrix_rank(data)
    if rank < data.shape[1]:
        LGR.critical(f"Regressor matrix is rank deficient: {rank}")

        data, regressor_positions = remove_collinear_columns(data, regressor_positions)
        LGR.critical(f"New regressor names and positions: {regressor_positions}")

    # To stop errors and warnings
    drop_columns = np.where(np.var(data, axis=0) == 0)[0].tolist()
    if drop_columns:
        col_names = [get_col_name(indx, regressor_positions) for indx in drop_columns]

        LGR.critical(f"The following have constant variance: {col_names}")

        data, regressor_positions = get_new_matrix_and_names(
            drop_columns, data, regressor_positions
        )

        LGR.critical(f"New regressor names and positions: {regressor_positions}")

    mean = data[censor_mask.astype(bool)].mean(axis=0)
    std = data[censor_mask.astype(bool)].std(axis=0, ddof=1)
    std[std < np.finfo(np.float64).eps] = 1.0
    data[censor_mask.astype(bool)] = (data[censor_mask.astype(bool)] - mean) / std

    np.savetxt(regressor_file, data, fmt="%.6f")

    return regressor_file


def save_event_file(timing_dir, trial_type, timing_data):
    filename = timing_dir / f"{trial_type}.1D"
    timing_str = " ".join(timing_data)
    with open(filename, "w") as f:
        f.write(timing_str)


def create_timing_files(subject_dir, event_file, task):
    timing_dir = subject_dir / "timing_files" / task
    timing_dir.mkdir(parents=True, exist_ok=True)

    event_df = pd.read_csv(event_file, sep="\t")
    trial_types = event_df["trial_type"].unique()

    for trial_type in trial_types:
        trial_df = event_df[event_df["trial_type"] == trial_type]
        row_mask = (
            np.full(len(trial_df), True, dtype=bool)
            if task != "flanker"
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

    # Get errors
    if task == "flanker":
        timing_data = trial_df.loc[~row_mask, "onset"].astype(str)

        save_event_file(timing_dir, trial_type="errors", timing_data=timing_data)

    return timing_dir
