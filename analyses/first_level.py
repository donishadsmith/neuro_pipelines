import argparse, json, subprocess, sys
from pathlib import Path

import bids, numpy as np, pandas as pd

from nifti2bids._helpers import iterable_to_str
from nifti2bids.logging import setup_logger
from nifti2bids.qc import compute_n_dummy_scans, create_censor_mask

from _utils import create_contrast_files

LGR = setup_logger(__name__)
LGR.setLevel("INFO")


def _get_cmd_args():
    parser = argparse.ArgumentParser(description="Perform first level GLM for a task.")
    parser.add_argument(
        "--bids_dir", dest="bids_dir", required=True, help="Path to BIDS directory."
    )
    parser.add_argument(
        "--afni_img_path",
        dest="afni_img_path",
        required=True,
        help="Path to Apptainer image of Afni with R.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination (output) directory.",
    )
    parser.add_argument(
        "--deriv_dir",
        dest="deriv_dir",
        required=False,
        default=None,
        help="Root of the derivatives directory.",
    )
    parser.add_argument(
        "--space",
        dest="space",
        default="MNIPediatricAsym_cohort-1_res-2",
        required=False,
        help="Template space.",
    )
    parser.add_argument(
        "--subject",
        dest="subject",
        required=True,
        help="Subject ID without the 'sub-' entity.",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--n_motion_parameters",
        dest="n_motion_parameters",
        default=24,
        type=int,
        required=False,
        help=(
            "Number of motion parameters to use: 6 (base trans + rot), "
            "12 (base + derivatives), 24 (base + derivatives + power)"
        ),
    )
    parser.add_argument(
        "--fd",
        dest="fd",
        default=0.9,
        type=float,
        required=False,
        help="Framewise displacement threshold.",
    )
    parser.add_argument(
        "--exclusion_criteria",
        dest="exclusion_criteria",
        default=0.40,
        type=float,
        required=False,
        help=(
            "Threshold to exclude entire run if more than the specified "
            "percentage of steady-state volumes are removed."
        ),
    )
    parser.add_argument(
        "--n_dummy_scans",
        dest="n_dummy_scans",
        default="auto",
        required=False,
        help=(
            "Number of dummy scans to remove. If 'auto' computes number of dummy scans "
            "by the numnber of 'non_steady_state_outlier_XX' columns."
        ),
    )
    parser.add_argument(
        "--n_acompcor",
        dest="n_acompcor",
        default=5,
        type=int,
        required=False,
        help="Number of aCompCor components.",
    )
    parser.add_argument(
        "--remove_global_signal",
        dest="remove_global_signal",
        default=True,
        required=False,
        help=(
            "Global signal regression. If True, ``n_motion_regressors`` is used "
            "to include derivative if 16, and power if 24"
        ),
    )
    parser.add_argument(
        "--fwhm",
        dest="fwhm",
        default=6,
        type=int,
        required=False,
        help="Spatial blurring.",
    )

    return parser


def get_acompcor_component_names(confounds_json_data, n_components):
    c_compcors = sorted([k for k in confounds_json_data.keys() if "c_comp_cor" in k])
    w_compcors = sorted([k for k in confounds_json_data.keys() if "w_comp_cor" in k])

    CSF = [c for c in c_compcors if confounds_json_data[c].get("Mask") == "CSF"][
        :n_components
    ]
    WM = [c for c in w_compcors if confounds_json_data[c].get("Mask") == "WM"][
        :n_components
    ]

    components_list = CSF + WM
    LGR.info(f"The following acompcor components will be used: {components_list}")

    return components_list


def get_motion_regressors(confounds_df, n_motion_parameters):
    motion_params = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    if n_motion_parameters in [12, 24]:
        derivatives = [f"{param}_derivative1" for param in motion_params]
        motion_params += derivatives

    if n_motion_parameters == 24:
        power = [f"{param}_power2" for param in motion_params]
        motion_params += power

    LGR.info(f"Using motion parameters: {motion_params}")

    return confounds_df[motion_params].to_numpy(copy=True)


def get_global_signal_regressors(confounds_df, n_motion_parameters):
    global_params = ["global_signal"]
    if n_motion_parameters in [12, 24]:
        global_params += ["global_signal_derivative1"]

    if n_motion_parameters == 24:
        global_params += ["global_signal_power2"]

    LGR.info(f"Using global signal parameters: {global_params}")

    return confounds_df[global_params].to_numpy(copy=True)


def create_censor_file(subject_dir, censor_mask):
    censor_file = subject_dir / "censor.1D"
    np.savetxt(censor_file, censor_mask, fmt="%d")

    return censor_file


def create_regressor_file(subject_dir, censor_mask, *regressor_arrays):
    regressor_file = subject_dir / "regressors.1D"
    valid_arrays = [arr for arr in regressor_arrays if arr is not None]
    data = np.column_stack(valid_arrays)

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

        if task == "flanker":
            timing_data = (
                trial_df.loc[
                    row_mask,
                    "onset",
                ]
                .astype(str)
                .to_numpy(copy=True)
            )
        else:
            timing_data = (
                trial_df.loc[
                    row_mask,
                    "onset",
                ].astype(str)
                + ":"
                + trial_df.loc[row_mask, "duration"].astype(str).to_numpy(copy=True)
            )

        if isinstance(timing_data, pd.Series):
            timing_data = timing_data.to_list()

        save_event_file(timing_dir, trial_type, timing_data)

    # Get errors
    if task == "flanker":
        timing_data = trial_df.loc[~row_mask, "onset"].astype(str)

        save_event_file(timing_dir, trial_type="errors", timing_data=timing_data)

    return timing_dir


def create_standardized_nifti_file(subject_dir, nifti_file, mask_file, censor_file):
    mean_file = subject_dir / Path(nifti_file).name.replace("preproc_bold", "mean")
    stdev_file = subject_dir / Path(nifti_file).name.replace("preproc_bold", "std")
    standardized_nifti_file = subject_dir / Path(nifti_file).name.replace(
        "preproc_bold", "standardized"
    )
    if not standardized_nifti_file.exists():
        censor_data = np.loadtxt(censor_file)
        kept_indices = np.where(censor_data == 1)[0]
        selector = ",".join(map(str, kept_indices))
        cmd_mean = (
            f"3dTstat -prefix {mean_file} "
            f"-mask {mask_file} "
            "-mean "
            f"-overwrite "
            f"'{nifti_file}[{selector}]'"
        )
        subprocess.run(cmd_mean, shell=True, check=True)
        cmd_std = (
            f"3dTstat -prefix {stdev_file} "
            f"-mask {mask_file} "
            "-stdev "
            f"-overwrite "
            f"'{nifti_file}[{selector}]'"
        )
        subprocess.run(cmd_std, shell=True, check=True)
        cmd_calc = (
            f"3dcalc -a {nifti_file} -b {mean_file} -c {stdev_file} -d {mask_file} "
            f"-expr 'd * (a - b) / c' -prefix {standardized_nifti_file} -overwrite "
        )
        subprocess.run(cmd_calc, shell=True, check=True)

    return standardized_nifti_file


def perform_spatial_smoothing(subject_dir, afni_img_path, nifti_file, mask_file, fwhm):
    smoothed_nifti_file = subject_dir / str(nifti_file).replace(
        "standardized", "smoothed"
    )

    if not smoothed_nifti_file.exists():
        cmd = (
            f"apptainer exec -B /projects:/projects {afni_img_path} 3dBlurToFWHM "
            f"-input {nifti_file} "
            f"-mask {mask_file} "
            f"-FWHM {fwhm} "
            f"-prefix {smoothed_nifti_file} "
            "-overwrite"
        )
        LGR.info(f"Performing spatial smoothing with fwhm={fwhm}: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

    return smoothed_nifti_file


def get_task_contrast_cmd(task, timing_dir, regressors_file):
    # Using stim_times_AM1 and dmUBLOCK so that duration doesn't need to passed
    # and is instead paired with the onset time for block designs
    # Likely overkill since duration variation is minimal, maybe change to
    # block and a fixed value and remove the durations later

    """
    https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/statistics/deconvolve_block.html
    The second column shows dmUBLOCK, with either a positive parameter (\geq 0) or no parameter.
    For arguments \geq 1, this function behaves exactly like dmBLOCK above. When the argument is
    0 or no parameter is given, then the response is similar to that of dmBLOCK in that
    the response amplitude varies, but different to it in that the scaling is such that
    convolved plateau height is scaled to unity, and short duration events are shorter than 1.
    """
    if task == "nback":
        contrast_cmd = {
            "num_stimts": "-num_stimts 4 ",
            "contrasts": f"-stim_times_AM1 1 {timing_dir / 'instruction.1D'} 'dmUBLOCK' -stim_label 1 instruction "
            f"-stim_times_AM1 2 {timing_dir / '0-back.1D'} 'dmUBLOCK' -stim_label 2 0-back "
            f"-stim_times_AM1 3 {timing_dir / '1-back.1D'} 'dmUBLOCK' -stim_label 3 1-back "
            f"-stim_times_AM1 4 {timing_dir / '2-back.1D'} 'dmUBLOCK' -stim_label 4 2-back "
            f"-ortvec {regressors_file} Nuisance "
            "-gltsym 'SYM: +1*1-back -1*0-back' -glt_label 1 1-back_vs_0-back "
            "-gltsym 'SYM: +1*2-back -1*0-back' -glt_label 2 2-back_vs_0-back "
            "-gltsym 'SYM: +1*2-back -1*1-back' -glt_label 3 2-back_vs_1-back ",
        }
    elif task == "mtle":
        contrast_cmd = {
            "num_stimts": "-num_stimts 2 ",
            "contrasts": f"-stim_times_AM1 1 {timing_dir / 'instruction.1D'} 'dmUBLOCK' -stim_label 1 instruction "
            f"-stim_times_AM1 2 {timing_dir / 'indoor.1D'} 'dmUBLOCK' -stim_label 2 indoor "
            f"-ortvec {regressors_file} Nuisance "
            "-gltsym 'SYM: +1*indoor' -glt_label 1 indoor ",
        }
    elif task == "mtlr":
        contrast_cmd = {
            "num_stimts": "-num_stimts 2 ",
            "contrasts": f"-stim_times_AM1 1 {timing_dir / 'instruction.1D'} 'dmUBLOCK' -stim_label 1 instruction "
            f"-stim_times_AM1 2 {timing_dir / 'seen.1D'} 'dmUBLOCK' -stim_label 2 seen "
            f"-ortvec {regressors_file} Nuisance "
            "-gltsym 'SYM: +1*seen' -glt_label 1 seen ",
        }
    elif task == "princess":
        contrast_cmd = {
            "num_stimts": "-num_stimts 2 ",
            "contrasts": f"-stim_times_AM1 1 {timing_dir / 'switch.1D'} 'dmUBLOCK' -stim_label 1 switch "
            f"-stim_times_AM1 2 {timing_dir / 'nonswitch.1D'} 'dmUBLOCK' -stim_label 2 nonswitch "
            f"-ortvec {regressors_file} Nuisance "
            "-gltsym 'SYM: +1*switch -1*nonswitch' -glt_label 1 switch_vs_nonswitch ",
        }
    else:
        # Note: simply multiply the coefficient image by -1 to get the opposite contast
        contrast_cmd = _create_flanker_contrast(timing_dir, regressors_file)

    return contrast_cmd


def _create_flanker_contrast(timing_dir, regressors_file):
    # Dynamically create the flanker contrast to avoid including contrasts that
    # have no data
    contrast_cmd = {
        "num_stimts": "-num_stimts {num_labels} ",
        "contrasts": "{stims} -ortvec {regressors_file} Nuisance {gltsyms}",
    }

    labels_dict = {
        "stims": (
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} congruent ",
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} incongruent ",
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} nogo ",
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} neutral ",
            "-stim_times {label} {timing_file} 'GAM' -stim_label {label} errors ",
        ),
        "gltsyms": (
            "-gltsym 'SYM: +1*congruent -1*neutral' -glt_label {label} congruent_vs_neutral ",
            "-gltsym 'SYM: +1*incongruent -1*neutral' -glt_label {label} incongruent_vs_neutral ",
            "-gltsym 'SYM: +1*nogo -1*neutral' -glt_label {label} nogo_vs_neutral ",
            "-gltsym 'SYM: +1*congruent -1*incongruent' -glt_label {label} congruent_vs_incongruent ",
            "-gltsym 'SYM: +1*congruent -1*nogo' -glt_label {label} congruent_vs_nogo ",
            "-gltsym 'SYM: +1*incongruent -1*nogo' -glt_label {label} incongruent_vs_nogo ",
        ),
    }

    files = ["congruent.1D", "incongruent.1D", "nogo.1D", "neutral.1D", "errors.1D"]
    empty_mask = np.array(
        [np.loadtxt(timing_dir / file, delimiter=" ").size == 0 for file in files]
    )

    nonempty_files = np.array(files)[~empty_mask]
    keep_trial_types = [file.removesuffix(".1D") for file in nonempty_files]

    # Length of the stims
    contrast_cmd["num_stimts"] = contrast_cmd["num_stimts"].format(
        num_labels=len(nonempty_files)
    )

    # Only keep stims without empty files
    stims = ""
    for label, trial_type in enumerate(keep_trial_types, start=1):
        bool_list = [
            trial_type == stim_string.rstrip().split(" ")[-1]
            for stim_string in labels_dict["stims"]
        ]

        stim_string = labels_dict["stims"][bool_list.index(True)]
        if label == len(keep_trial_types):
            stim_string = stim_string.rstrip()

        stims += stim_string.format(
            label=label, timing_file=timing_dir / f"{trial_type}.1D"
        )

    # Only keep gltsym with two
    kept_gltsyms = []
    for gltsym in labels_dict["gltsyms"]:
        glt_label = gltsym.rstrip().split(" ")[-1]
        glt_label_parts = glt_label.split("_vs_")
        if all(
            glt_label_part in keep_trial_types for glt_label_part in glt_label_parts
        ):
            kept_gltsyms.append(gltsym)

    gltsyms = ""
    for label, gltsym in enumerate(kept_gltsyms, start=1):
        gltsyms += gltsym.format(label=label)

    contrast_cmd["contrasts"] = contrast_cmd["contrasts"].format(
        stims=stims, regressors_file=regressors_file, gltsyms=gltsyms
    )

    return contrast_cmd


def create_design_matrix(
    subject_dir, smoothed_nifti_file, mask_file, censor_file, contrast_cmd
):
    design_matrix_file = subject_dir / str(smoothed_nifti_file).replace(
        "smoothed.nii.gz", "design_matrix.1D"
    )

    cmd = (
        "3dDeconvolve "
        f"-input {smoothed_nifti_file} "
        f"-mask {mask_file} "
        f"-censor {censor_file} "
        "-polort 0 "
        "-local_times "
        f"{contrast_cmd['num_stimts']} "
        f"{contrast_cmd['contrasts']} "
        f"-x1D {design_matrix_file} "
        "-x1D_stop "
        "-overwrite"
    )

    LGR.info(f"Running 3dDeconvolve to create design matrix: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return design_matrix_file


def perform_first_level(
    subject_dir,
    afni_img_path,
    design_matrix_file,
    smoothed_nifti_file,
    mask_file,
):
    stats_file_relm = subject_dir / Path(smoothed_nifti_file).name.replace(
        "smoothed", "stats"
    )

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dREMLfit "
        f"-matrix {design_matrix_file} "
        f"-input {smoothed_nifti_file} "
        f"-mask {mask_file} "
        "-fout -tout "
        "-verb "
        f"-Rbuck {stats_file_relm} "
        "-overwrite"
    )

    LGR.info(
        f"Running 3dREMLfit for first level accounting for auto-correlation: {cmd}"
    )
    subprocess.run(cmd, shell=True, check=True)

    return stats_file_relm


def main(
    bids_dir,
    afni_img_path,
    dst_dir,
    deriv_dir,
    space,
    subject,
    task,
    n_motion_parameters,
    remove_global_signal,
    fd,
    exclusion_criteria,
    n_dummy_scans,
    n_acompcor,
    fwhm,
):
    tasknames = ["princess", "flanker", "nback", "mtle", "mtlr"]
    if task not in tasknames:
        LGR.critical(
            f"The task must be one of the following: {iterable_to_str(tasknames)}"
        )
        sys.exit()

    layout = bids.BIDSLayout(bids_dir, derivatives=deriv_dir or True)

    sessions = layout.get(
        subject=subject, task=task, target="session", return_type="id"
    )
    if not sessions:
        LGR.critical(f"No sessions for {subject} for {task}.")
        sys.exit()

    for session in sessions:
        confounds_tsv_files = layout.get(
            scope="derivatives",
            subject=subject,
            session=session,
            task=task,
            desc="confounds",
            extension="tsv",
            return_type="file",
        )
        if not confounds_tsv_files:
            continue
        else:
            confounds_tsv_file = confounds_tsv_files[0]

        confounds_json_file = layout.get(
            scope="derivatives",
            subject=subject,
            session=session,
            task=task,
            desc="confounds",
            extension="json",
            return_type="file",
        )
        if not confounds_json_file:
            continue
        else:
            confounds_json_file = confounds_json_file[0]

        event_file = layout.get(
            scope="raw",
            subject=subject,
            session=session,
            task=task,
            suffix="events",
            extension="tsv",
            return_type="file",
        )
        if not event_file:
            continue
        else:
            event_file = event_file[0]

        # Space parameter not getting template
        mask_files = layout.get(
            scope="derivatives",
            subject=subject,
            session=session,
            task=task,
            suffix="mask",
            extension="nii.gz",
            return_type="file",
        )
        if not mask_files:
            continue
        else:
            mask_file = [file for file in mask_files if space in str(Path(file).name)][
                0
            ]
            LGR.info(f"Using the following mask file: {mask_file}")

        nifti_files = layout.get(
            scope="derivatives",
            subject=subject,
            session=session,
            task=task,
            suffix="bold",
            extension="nii.gz",
            return_type="file",
        )
        if not nifti_files:
            continue
        else:
            nifti_file = [
                file for file in nifti_files if space in str(Path(file).name)
            ][0]
            LGR.info(f"Using the following mask file: {nifti_file}")

        # Create subject directory
        subject_dir = Path(dst_dir) / f"sub-{subject}" / f"ses-{session}" / "func"
        subject_dir.mkdir(parents=True, exist_ok=True)

        confounds_df = pd.read_csv(confounds_tsv_file, sep="\t").fillna(0)

        if n_dummy_scans == "auto":
            n_dummy_scans = compute_n_dummy_scans(confounds_df)
            LGR.info(f"There are {n_dummy_scans} non-steady state scans.")

        # Censor File
        censor_mask = create_censor_mask(
            confounds_df,
            column_name="framewise_displacement",
            n_dummy_scans=n_dummy_scans,
            threshold=fd,
        )

        # TODO: Incorporate exclusion criteria that is appropriate given the
        # demographics of sample
        kept = censor_mask[n_dummy_scans:]
        n_censored = np.sum(kept == 0)
        percent_censored = n_censored / kept.size
        LGR.critical(
            f"For SUBJECT: {subject}, SESSION: {session}, TASK: {task}, "
            f"proportion of steady state volumes removed at an fd > {fd} mm: "
            f" {percent_censored}"
        )

        if percent_censored > exclusion_criteria:
            LGR.critical(
                f"For SUBJECT: {subject}, SESSION: {session}, TASK: {task}, "
                "run excluded because the percent censored is greater than the "
                f"exclusion criteria: {exclusion_criteria}"
            )
            continue

        censor_file = create_censor_file(subject_dir, censor_mask)

        # Regressors
        with open(confounds_json_file, "r") as f:
            confounds_meta = json.load(f)

        motion_regressors = get_motion_regressors(confounds_df, n_motion_parameters)
        global_regressors = (
            get_global_signal_regressors(confounds_df, n_motion_parameters)
            if remove_global_signal
            else None
        )

        acompcor_names = get_acompcor_component_names(confounds_meta, n_acompcor)
        acompcor_regressors = confounds_df[acompcor_names].to_numpy(copy=True)

        regressors_file = create_regressor_file(
            subject_dir,
            censor_mask,
            motion_regressors,
            acompcor_regressors,
            global_regressors,
        )

        # Create timing files
        timing_dir = create_timing_files(subject_dir, event_file, task)

        # Z-score data
        standardized_nifti_file = create_standardized_nifti_file(
            subject_dir, nifti_file, mask_file, censor_file
        )

        # Smooth data
        smoothed_nifti_file = perform_spatial_smoothing(
            subject_dir, afni_img_path, standardized_nifti_file, mask_file, fwhm
        )

        # Create design matrix
        contrast_cmd = get_task_contrast_cmd(task, timing_dir, regressors_file)
        design_matrix_file = create_design_matrix(
            subject_dir,
            smoothed_nifti_file,
            mask_file,
            censor_file,
            contrast_cmd,
        )

        # Perform first level
        stats_file_relm = perform_first_level(
            subject_dir,
            afni_img_path,
            design_matrix_file,
            smoothed_nifti_file,
            mask_file,
        )

        contrast_dir = stats_file_relm.parent / "contrasts"
        if not contrast_dir.exists():
            contrast_dir.mkdir()

        create_contrast_files(stats_file_relm, contrast_dir, afni_img_path, task)


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
