import argparse, json, sys
from pathlib import Path

import bids, numpy as np, pandas as pd

from nifti2bids._helpers import iterable_to_str
from nifti2bids.logging import setup_logger
from nifti2bids.qc import compute_n_dummy_scans, create_censor_mask

from _denoising import (
    get_acompcor_component_names,
    get_cosine_regressors,
    get_global_signal_regressors,
    get_motion_regressors,
    percent_signal_change,
    perform_spatial_smoothing,
)
from _gen_afni_files import (
    create_censor_file,
    create_timing_files,
    create_nuisance_regressor_file,
)
from _models import create_design_matrix, perform_first_level
from _utils import create_beta_files

LGR = setup_logger(__name__)


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
        default=12,
        type=int,
        choices=[6, 12, 18, 24],
        required=False,
        help=(
            "Number of motion parameters to use: 6 (base trans + rot), "
            "12 (base + derivatives), 18 (base + derivatives + power), "
            "24 (base + derivatives + power + derivative power)."
        ),
    )
    parser.add_argument(
        "--fd_threshold",
        dest="fd_threshold",
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
        "--acompcor_strategy",
        dest="acompcor_strategy",
        default="combined",
        choices=["combined", "separate"],
        required=False,
        help="Whether to use combined aCompCor components or 'separate' components.",
    )
    parser.add_argument(
        "--n_global_parameters",
        dest="n_global_parameters",
        default=1,
        choices=[0, 1, 2, 3, 4],
        required=False,
        help=(
            "Global signal regression. If 0, no global signal parameters used. "
            "If 1, 'global_signal' is used, if 2 'global_signal' and 'global_signal_derivative1' used "
            "If 3, 'global_signal', global_signal_derivative1', and global_signal_power2' used. "
            "If 4, 'global_signal', global_signal_derivative1', global_signal_power2', an global_signal_derivative1_power2' used."
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


def get_task_deconvolve_cmd(task, timing_dir, nuisance_regressors_file):
    if task == "nback":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 4 ",
            "args": f"-stim_times 1 {timing_dir / 'instruction.1D'} 'BLOCK(2, 1)' -stim_label 1 instruction "
            f"-stim_times 2 {timing_dir / '0-back.1D'} 'BLOCK(32, 1)' -stim_label 2 0-back "
            f"-stim_times 3 {timing_dir / '1-back.1D'} 'BLOCK(32, 1)' -stim_label 3 1-back "
            f"-stim_times 4 {timing_dir / '2-back.1D'} 'BLOCK(32, 1)' -stim_label 4 2-back "
            f"-ortvec {nuisance_regressors_file} Nuisance "
            "-gltsym 'SYM: +1*1-back -1*0-back' -glt_label 1 1-back_vs_0-back "
            "-gltsym 'SYM: +1*2-back -1*0-back' -glt_label 2 2-back_vs_0-back "
            "-gltsym 'SYM: +1*2-back -1*1-back' -glt_label 3 2-back_vs_1-back ",
        }
    elif task == "mtle":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 2 ",
            "args": f"-stim_times 1 {timing_dir / 'instruction.1D'} 'BLOCK(2, 1)' -stim_label 1 instruction "
            f"-stim_times 2 {timing_dir / 'indoor.1D'} 'BLOCK(18, 1)' -stim_label 2 indoor "
            f"-ortvec {nuisance_regressors_file} Nuisance ",
        }
    elif task == "mtlr":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 2 ",
            "args": f"-stim_times 1 {timing_dir / 'instruction.1D'} 'BLOCK(2, 1)' -stim_label 1 instruction "
            f"-stim_times 2 {timing_dir / 'seen.1D'} 'BLOCK(18, 1)' -stim_label 2 seen "
            f"-ortvec {nuisance_regressors_file} Nuisance ",
        }
    elif task == "princess":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 2 ",
            "args": f"-stim_times 1 {timing_dir / 'switch.1D'} 'BLOCK(52, 1)' -stim_label 1 switch "
            f"-stim_times 2 {timing_dir / 'nonswitch.1D'} 'BLOCK(52, 1)' -stim_label 2 nonswitch "
            f"-ortvec {nuisance_regressors_file} Nuisance "
            "-gltsym 'SYM: +1*switch -1*nonswitch' -glt_label 1 switch_vs_nonswitch ",
        }
    else:
        # Note: simply multiply the coefficient image by -1 to get the opposite contrast
        deconvolve_cmd = create_flanker_deconvolve_cmd(
            timing_dir, nuisance_regressors_file
        )

    return deconvolve_cmd


def create_flanker_deconvolve_cmd(timing_dir, nuisance_regressors_file):
    # Dynamically create the flanker contrast to avoid including contrasts that
    # have no data
    deconvolve_cmd = {
        "num_stimts": "-num_stimts {num_labels} ",
        "args": "{stims} -ortvec {nuisance_regressors_file} Nuisance {gltsyms}",
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
    deconvolve_cmd["num_stimts"] = deconvolve_cmd["num_stimts"].format(
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

        stims += stim_string.format(
            label=label, timing_file=timing_dir / f"{trial_type}.1D"
        )

    stims = stims.rstrip()

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

    gltsyms = gltsyms.rstrip()

    deconvolve_cmd["args"] = deconvolve_cmd["args"].format(
        stims=stims, nuisance_regressors_file=nuisance_regressors_file, gltsyms=gltsyms
    )

    return deconvolve_cmd


def main(
    bids_dir,
    afni_img_path,
    dst_dir,
    deriv_dir,
    space,
    subject,
    task,
    n_motion_parameters,
    n_global_parameters,
    fd_threshold,
    exclusion_criteria,
    n_dummy_scans,
    n_acompcor,
    acompcor_strategy,
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

        subject_dir = Path(dst_dir) / f"sub-{subject}" / f"ses-{session}" / "func"
        subject_dir.mkdir(parents=True, exist_ok=True)

        confounds_df = pd.read_csv(confounds_tsv_file, sep="\t").fillna(0)

        if n_dummy_scans == "auto":
            n_dummy_scans = compute_n_dummy_scans(confounds_df)
            LGR.info(f"There are {n_dummy_scans} non-steady state scans.")

        censor_mask = create_censor_mask(
            confounds_df,
            column_name="framewise_displacement",
            n_dummy_scans=n_dummy_scans,
            threshold=fd_threshold,
        )

        kept = censor_mask[n_dummy_scans:]
        n_censored = np.sum(kept == 0)
        percent_censored = n_censored / kept.size
        LGR.critical(
            f"For SUBJECT: {subject}, SESSION: {session}, TASK: {task}, "
            f"proportion of steady state volumes removed at an fd threshold > {fd_threshold} mm: "
            f" {percent_censored}"
        )

        if percent_censored > exclusion_criteria:
            LGR.critical(
                f"For SUBJECT: {subject}, SESSION: {session}, TASK: {task}, "
                "run excluded because the percent censored is greater than the "
                f"exclusion criteria: {exclusion_criteria}"
            )
            continue

        censor_file = create_censor_file(
            subject_dir, subject, session, task, space, censor_mask
        )

        with open(confounds_json_file, "r") as f:
            confounds_meta = json.load(f)

        cosine_regressors, cosine_regressor_names = get_cosine_regressors(confounds_df)

        motion_regressors, motion_regressor_names = get_motion_regressors(
            confounds_df, n_motion_parameters
        )

        acompcor_regressor_names = get_acompcor_component_names(
            confounds_meta, n_acompcor, acompcor_strategy
        )
        acompcor_regressors = confounds_df[acompcor_regressor_names].to_numpy(copy=True)

        global_regressors, global_regressor_names = (
            get_global_signal_regressors(confounds_df, n_global_parameters)
            if n_global_parameters
            else (None, None)
        )

        regressor_names_nested_list = filter(
            None,
            [
                cosine_regressor_names,
                motion_regressor_names,
                acompcor_regressor_names,
                global_regressor_names,
            ],
        )
        regressor_names = [
            regressor
            for regressor_list in regressor_names_nested_list
            for regressor in regressor_list
        ]
        nuisance_regressors_file = create_nuisance_regressor_file(
            subject_dir,
            subject,
            session,
            task,
            space,
            censor_mask,
            regressor_names,
            cosine_regressors,
            motion_regressors,
            acompcor_regressors,
            global_regressors,
        )

        timing_dir = create_timing_files(subject_dir, event_file, task)

        percent_change_nifti_file = percent_signal_change(
            subject_dir, afni_img_path, nifti_file, mask_file, censor_file
        )

        smoothed_nifti_file = perform_spatial_smoothing(
            subject_dir, afni_img_path, percent_change_nifti_file, mask_file, fwhm
        )

        deconvolve_cmd = get_task_deconvolve_cmd(
            task, timing_dir, nuisance_regressors_file
        )
        design_matrix_file = create_design_matrix(
            subject_dir,
            afni_img_path,
            smoothed_nifti_file,
            mask_file,
            censor_file,
            deconvolve_cmd,
            cosine_regressor_names,
        )

        stats_file_relm = perform_first_level(
            subject_dir,
            afni_img_path,
            design_matrix_file,
            smoothed_nifti_file,
            mask_file,
        )

        betas_dir = stats_file_relm.parent / "betas"
        betas_dir.mkdir(parents=True, exist_ok=True)

        create_beta_files(
            stats_file_relm, betas_dir, afni_img_path, task, analysis_type="glm"
        )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
