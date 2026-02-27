"""
gPPI workflow

Papers:
1) https://pmc.ncbi.nlm.nih.gov/articles/PMC4632075/
2) https://pmc.ncbi.nlm.nih.gov/articles/PMC3376181/

Forums:
1) https://discuss.afni.nimh.nih.gov/t/gppi-analysis-and-upsampling/172

AFNI:
1) https://afni.nimh.nih.gov/CD-CorrAna
2) https://web.archive.org/web/20241103095511/https://afni.nimh.nih.gov/CD-CorrAna
   (archived website)

1) Collect confounds and create censor file
2) PSC scaling of NIfTI image, compute mean for censored files
3) Resample mask to NIfTI (if needed) then extract timeseries
4) Tranpose the seed timeseries to a column vector
5) Denoise seed timeseries not too aggressively. Note that smoothing is not done prior
to extracting the seed, the timeseries is already averaged which helps with spatial noise
reduction. More importantly, smoothing blur signal outside of the voxels of interest into
the will result in your seed timeseries containing signal from voxels outside of your mask

For each condition in task (6-9):
6) Upsample seed timeseries and task regressor to 0.1
   (TR_orig/ TR_sub is equal to number of points added between each TR or
   the duration / TR_sub is equal to the number of points added after each onset
   time)
7) Deconvolve seed timeseries to get the neural signal that will later
   interact with the task regressor and this interaction will be convolved.
8) Create PPI term PPI = ([neural signal * binary_condition_vector] * hrf)(t).
   Use GAM.
9) Downsample the PPI term back down to the true TR grid

After:
10) For NIfTI image, smooth, then use 3ddeconvolve. Ensure to model everything
   from nuisance regressors, all main effect conditions (convolved), the
   denoised seed signal, and the PPI interaction terms (already convolved
   in previous step). Create contrasts of the interaction terms (+ means
   greater connectivity for A than B and - means reduced connectivity for
   A relative to B)
11) Use 3dremlfit to account for temporal autocorrelation
12) Extract PPI interaction contrasts betas for downstream analyses
"""

import argparse, subprocess, json, subprocess, sys
from pathlib import Path

import nibabel as nib, numpy as np

import bids, numpy as np, pandas as pd

from nifti2bids._helpers import iterable_to_str
from nifti2bids.logging import setup_logger
from nifti2bids.metadata import get_tr, get_n_volumes
from nifti2bids.qc import (
    compute_n_dummy_scans,
    create_censor_mask,
)

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
    is_timing_file_empty,
)
from _argparse_typing import n_dummy_type
from _models import create_design_matrix, perform_first_level
from _utils import (
    create_beta_files,
    get_beta_names,
    get_coordinate_from_filename,
    get_first_level_gltsym_codes,
    resample_seed_img,
)

LGR = setup_logger(__name__)

# Using constant durations instead of BIDS one, which have small
# stimulus presentation delays
# Instruction has the same duration for all three tasks but in the
# code for clarity
CONDITION_DURATIONS = {
    "flanker": 0.5,
    "nback": 32,
    "princess": 52,
    "mtle": 18,
    "mtlr": 18,
    "instruction_nback": 2,
    "instruction_mtle": 2,
    "instruction_mtlr": 2,
}


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Perform first level gPPI (task-based functional connectivty) for a task."
    )
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
        "--scratch_dir",
        dest="scratch_dir",
        required=True,
        help="Path to the scratch directory.",
    )
    parser.add_argument(
        "--deriv_dir",
        dest="deriv_dir",
        required=False,
        default=None,
        help="Root of the derivatives directory.",
    )
    parser.add_argument(
        "--seed_mask_path",
        dest="seed_mask_path",
        required=True,
        help="The mask of the seed region.",
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
            "24 (base + derivatives + power + derivative power). "
            "Seed denoising will always exclusively use 6 motion parameters (base trans + rot)"
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
        type=n_dummy_type,
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
        type=str,
        required=False,
        help="Whether to use combined aCompCor components or 'separate' components.",
    )
    parser.add_argument(
        "--n_global_parameters",
        dest="n_global_parameters",
        default=1,
        choices=[0, 1, 2, 3, 4],
        type=int,
        required=False,
        help=(
            "Global signal regression. If 0, no global signal parameters used. "
            "If 1, 'global_signal' is used, if 2 'global_signal' and 'global_signal_derivative1' used "
            "If 3, 'global_signal', global_signal_derivative1', and global_signal_power2' used. "
            "If 4, 'global_signal', global_signal_derivative1', global_signal_power2', an global_signal_derivative1_power2' used. "
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
    parser.add_argument(
        "--upsample_dt",
        dest="upsample_dt",
        default=0.1,
        type=float,
        required=False,
        help="Time resolution to upsample seed timeseries (and condition times) to prior to deconvolution ",
    )
    parser.add_argument(
        "--faltung_penalty_syntax",
        dest="faltung_penalty_syntax",
        default="012 0",
        required=False,
        type=str,
        help=(
            "Deconvolution penalty syntax to pass to the FALTUNG parameter in 3dTfitter "
            "(fset fpre pen fac). See: https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTfitter.html"
        ),
    )

    return parser


def extract_seed_timeseries(
    subject_analysis_dir,
    subject_scratch_dir,
    subject_nifti_file,
    seed_mask_path,
    afni_img_path,
):
    LGR.info(f"Using the following seed mask file: {seed_mask_path}")

    possible_coordinate = get_coordinate_from_filename(
        seed_mask_path, replace_underscore=False
    )
    if possible_coordinate:
        seed_name = f"seed_{possible_coordinate}"
    else:
        seed_name = f"seed"

    seed_timeseries_file = (
        subject_analysis_dir / "seed" / f"{seed_name}_desc-timeseries.1D"
    )
    seed_timeseries_file.parent.mkdir(parents=True, exist_ok=True)

    seed_img = resample_seed_img(nib.load(seed_mask_path), nib.load(subject_nifti_file))

    resampled_seed_file = subject_scratch_dir / f"resampled_{seed_mask_path.name}"
    nib.save(seed_img, resampled_seed_file)

    # Note: output is a column vector
    cmd = (
        f"apptainer exec -B /projects:/projects -B /scratch:/scratch {afni_img_path} 3dmaskave "
        f"-mask {resampled_seed_file} "
        f"-q {subject_nifti_file} > {seed_timeseries_file}"
    )

    LGR.info(f"Extracting seed: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return seed_timeseries_file


def denoise_seed_timeseries(
    subject_analysis_dir,
    subject,
    session,
    task,
    space,
    seed_timeseries_file,
    censor_file,
    afni_img_path,
    confounds_df,
):

    denoised_seed_timeseries_file = (
        seed_timeseries_file.parent
        / seed_timeseries_file.name.replace("_desc-timeseries", "_desc-denoised")
    )
    regressor_names = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    motion_regressors = confounds_df[regressor_names].to_numpy()

    LGR.info(
        "Only the six base motion parameters will be used to denoise the "
        f"seed timeseries: {regressor_names}"
    )

    censor_mask = np.loadtxt(censor_file)
    seed_nuisance_regressor_file = create_nuisance_regressor_file(
        subject_analysis_dir,
        subject,
        session,
        task,
        space,
        censor_mask,
        regressor_names,
        motion_regressors,
        regressor_file_prefix="seed",
    )

    # Note: Some Afni functions only accept rows and require \', using \\' to
    # make the backslash literal
    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dTproject "
        f"-input {seed_timeseries_file}\\' "
        f"-ort {seed_nuisance_regressor_file} "
        f"-polort A "
        f"-censor {censor_file} "
        "-cenmode ZERO "
        f"-prefix {denoised_seed_timeseries_file}"
    )

    LGR.info(f"Denoising seed: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return denoised_seed_timeseries_file


def get_task_deconvolve_cmd(
    task, timing_dir, nuisance_regressors_file, seed_timeseries_file, ppi_dir
):
    seed_name = str(seed_timeseries_file).split("_desc")[0]

    if task == "nback":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 9 ",
            "args": f"-stim_file 1 {seed_timeseries_file} -stim_label 1 {seed_name} "
            f"-stim_times 2 {timing_dir / 'instruction.1D'} 'BLOCK(2, 1)' -stim_label 2 instruction "
            f"-stim_times 3 {timing_dir / '0-back.1D'} 'BLOCK(32, 1)' -stim_label 3 0-back "
            f"-stim_times 4 {timing_dir / '1-back.1D'} 'BLOCK(32, 1)' -stim_label 4 1-back "
            f"-stim_times 5 {timing_dir / '2-back.1D'} 'BLOCK(32, 1)' -stim_label 5 2-back "
            f"-stim_file 6 {ppi_dir / 'PPI_instruction.1D'} -stim_label 6 PPI_instruction "
            f"-stim_file 7 {ppi_dir / 'PPI_0-back.1D'} -stim_label 7 PPI_0-back "
            f"-stim_file 8 {ppi_dir / 'PPI_1-back.1D'} -stim_label 8 PPI_1-back "
            f"-stim_file 9 {ppi_dir / 'PPI_2-back.1D'} -stim_label 9 PPI_2-back "
            f"-ortvec {nuisance_regressors_file} Nuisance "
            "-gltsym 'SYM: +1*PPI_1-back -1*PPI_0-back' -glt_label 1 PPI_1-back_vs_PPI_0-back "
            "-gltsym 'SYM: +1*PPI_2-back -1*PPI_0-back' -glt_label 2 PPI_2-back_vs_PPI_0-back "
            "-gltsym 'SYM: +1*PPI_2-back -1*PPI_1-back' -glt_label 3 PPI_2-back_vs_PPI_1-back ",
        }
    elif task == "mtle":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 5 ",
            "args": f"-stim_file 1 {seed_timeseries_file} -stim_label 1 {seed_name} "
            f"-stim_times 2 {timing_dir / 'instruction.1D'} 'BLOCK(2, 1)' -stim_label 2 instruction "
            f"-stim_times 3 {timing_dir / 'indoor.1D'} 'BLOCK(18, 1)' -stim_label 3 indoor "
            f"-stim_file 4 {ppi_dir / 'PPI_instruction.1D'} -stim_label 4 PPI_instruction "
            f"-stim_file 5 {ppi_dir / 'PPI_indoor.1D'} -stim_label 5 PPI_instruction "
            f"-ortvec {nuisance_regressors_file} Nuisance ",
        }
    elif task == "mtlr":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 5 ",
            "args": f"-stim_file 1 {seed_timeseries_file} -stim_label 1 {seed_name} "
            f"-stim_times 2 {timing_dir / 'instruction.1D'} 'BLOCK(2, 1)' -stim_label 2 instruction "
            f"-stim_times 3 {timing_dir / 'seen.1D'} 'BLOCK(18, 1)' -stim_label 3 seen "
            f"-stim_file 4 {ppi_dir / 'PPI_instruction.1D'} -stim_label 4 PPI_instruction "
            f"-stim_file 5 {ppi_dir / 'PPI_seen.1D'} -stim_label 5 PPI_seen "
            f"-ortvec {nuisance_regressors_file} Nuisance ",
        }
    elif task == "princess":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 5 ",
            "args": f"-stim_file 1 {seed_timeseries_file} -stim_label 1 {seed_name} "
            f"-stim_times 2 {timing_dir / 'switch.1D'} 'BLOCK(52, 1)' -stim_label 2 switch "
            f"-stim_times 3 {timing_dir / 'nonswitch.1D'} 'BLOCK(52, 1)' -stim_label 3 nonswitch "
            f"-stim_file 4 {ppi_dir / 'PPI_switch.1D'} -stim_label 4 PPI_switch "
            f"-stim_file 5 {ppi_dir / 'PPI_nonswitch.1D'} -stim_label 5 PPI_nonswitch "
            f"-ortvec {nuisance_regressors_file} Nuisance "
            "-gltsym 'SYM: +1*PPI_switch -1*PPI_nonswitch' -glt_label 1 PPI_switch_vs_PPI_nonswitch ",
        }
    else:
        # Note: simply multiply the coefficient image by -1 to get the opposite contrast
        deconvolve_cmd = create_flanker_deconvolve_cmd(
            timing_dir, nuisance_regressors_file, seed_timeseries_file, ppi_dir
        )

    return deconvolve_cmd


def create_flanker_deconvolve_cmd(
    timing_dir, nuisance_regressors_file, seed_timeseries_file, ppi_dir
):
    # Dynamically create the flanker contrast to avoid including contrasts that
    # have no data

    deconvolve_cmd = {
        "num_stimts": "-num_stimts {num_labels} ",
        "args": "{stims} -ortvec {nuisance_regressors_file} Nuisance {gltsyms}",
    }

    labels_dict = {
        "stims": (
            "-stim_times {time_label} {timing_file} 'GAM' -stim_label {label} congruent ",
            "-stim_times {time_label} {timing_file} 'GAM' -stim_label {label} incongruent ",
            "-stim_times {time_label} {timing_file} 'GAM' -stim_label {label} nogo ",
            "-stim_times {time_label} {timing_file} 'GAM' -stim_label {label} neutral ",
            "-stim_times {time_label} {timing_file} 'GAM' -stim_label {label} errors ",
            f"-stim_file {{ppi_file}} -stim_label {{label}} PPI_congruent ",
            f"-stim_file {{ppi_file}} -stim_label {{label}} PPI_incongruent ",
            f"-stim_file {{ppi_file}} -stim_label {{label}} PPI_nogo ",
            f"-stim_file {{ppi_file}} -stim_label {{label}} PPI_neutral ",
            f"-stim_file {{ppi_file}} -stim_label {{label}} PPI_errors ",
        ),
        "gltsyms": (
            "-gltsym 'SYM: +1*PPI_congruent -1*PPI_neutral' -glt_label {label} PPI_congruent_vs_PPI_neutral ",
            "-gltsym 'SYM: +1*PPI_incongruent -1*PPI_neutral' -glt_label {label} PPI_incongruent_vs_PPI_neutral ",
            "-gltsym 'SYM: +1*PPI_nogo -1*PPI_neutral' -glt_label {label} PPI_nogo_vs_PPI_neutral ",
            "-gltsym 'SYM: +1*PPI_incongruent -1*PPI_congruent' -glt_label {label} PPI_incongruent_vs_PPI_congruent ",
            "-gltsym 'SYM: +1*PPI_congruent -1*PPI_nogo' -glt_label {label} PPI_congruent_vs_PPI_nogo ",
            "-gltsym 'SYM: +1*PPI_incongruent -1*PPI_nogo' -glt_label {label} PPI_incongruent_vs_PPI_nogo ",
        ),
    }

    files = ["congruent.1D", "incongruent.1D", "nogo.1D", "neutral.1D", "errors.1D"]
    empty_mask = np.array([is_timing_file_empty(timing_dir / file) for file in files])

    nonempty_files = np.array(files)[~empty_mask]
    keep_trial_regressors = [file.removesuffix(".1D") for file in nonempty_files]
    keep_ppi_regressors = [
        f"PPI_{trial_regressor}" for trial_regressor in keep_trial_regressors
    ]
    keep_trial_regressors += keep_ppi_regressors

    # Only keep stims without empty files
    seed_name = str(seed_timeseries_file).split("_desc")[0]
    stims = f"-stim_file 1 {seed_timeseries_file} -stim_label 1 {seed_name} "
    for label, regressor in enumerate(keep_trial_regressors, start=2):
        bool_list = [
            regressor == stim_string.rstrip().split(" ")[-1]
            for stim_string in labels_dict["stims"]
        ]

        stim_string = labels_dict["stims"][bool_list.index(True)]

        if "PPI_" in stim_string:
            stims += stim_string.format(
                label=label, ppi_file=ppi_dir / f"{regressor}.1D"
            )
        else:
            stims += stim_string.format(
                time_label=label,
                label=label,
                timing_file=timing_dir / f"{regressor}.1D",
            )

    stims = stims.rstrip()

    # Length of the stims
    deconvolve_cmd["num_stimts"] = deconvolve_cmd["num_stimts"].format(
        num_labels=label - 1
    )

    # Only keep gltsym with two
    kept_gltsyms = []
    for gltsym in labels_dict["gltsyms"]:
        glt_label = gltsym.rstrip().split(" ")[-1]
        glt_label_parts = glt_label.split("_vs_")
        if all(
            glt_label_part in keep_ppi_regressors for glt_label_part in glt_label_parts
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


def get_instruction_name(timing_dir, task, condition_filenames):
    if task in ["nback", "mtle", "mtlr"]:
        return condition_filenames + [timing_dir / f"instruction.1D"]
    else:
        return condition_filenames


def resample_data(target_file, tr, afni_img_path, upsample_dt, method):
    if method == "upsample":
        resampled_filename = target_file.parent / target_file.name.replace(
            "_desc-denoised", "_desc-upsampled"
        )

        # New length of interpolated timseries is (tr / upsample_dt) * n_original_volumes
        cmd = (
            f"apptainer exec -B /projects:/projects {afni_img_path} 1dUpsample {int(tr / upsample_dt)} "
            f"{str(target_file)}\\' > {resampled_filename}"
        )

        LGR.info(f"Upsampling seed timeseries from {tr} to {upsample_dt}: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    else:
        # original TR divided by sub_TR, starts at the first tr (0) and takes every
        # (tr / upsample_dt) point
        resampled_filename = (
            target_file.parent
            / f"PPI_{target_file.name.replace('_desc-PPI_upsampled.1D', '.1D')}"
        )
        cmd = (
            f"apptainer exec -B /projects:/projects {afni_img_path} 1dcat "
            f"'{target_file}{{0..$({int(tr / upsample_dt)})}}' > {resampled_filename}"
        )

        LGR.info(
            f"Downsampling the PPI regressor back to the original {tr} s grid: {cmd}"
        )
        subprocess.run(cmd, shell=True, check=True)

    return resampled_filename


def deconvolve_seed_timeseries(
    upsampled_seed_timeseries_file, upsample_dt, faltung_penalty_syntax, afni_img_path
):
    gamma_file_name = upsampled_seed_timeseries_file.parent / "GammaHR.1D"
    deconvolved_seed_timeseries_file = (
        upsampled_seed_timeseries_file.parent
        / upsampled_seed_timeseries_file.name.replace(
            "_desc-upsampled", "_desc-deconvolved"
        )
    )

    # Create impulse response function (GAM) and perform deconvolution to estimate the neural response given the
    # upsampled seed timeseries and an impulse response function, while also adding a penalty for better/smoother
    # estimation
    cmd = (
        f'apptainer exec -B /projects:/projects {afni_img_path} bash -c "waver '
        f"-dt {upsample_dt} -GAM -inline 1@1 > {gamma_file_name} && "
        f"3dTfitter -RHS {upsampled_seed_timeseries_file} "
        f'-FALTUNG {gamma_file_name} {deconvolved_seed_timeseries_file} {faltung_penalty_syntax}"'
    )

    LGR.info(f"Deconvolving upsampled seed timeseries: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return deconvolved_seed_timeseries_file


def upsample_condition_regressor(
    timing_file, task, tr, n_volumes, upsample_dt, afni_img_path
):
    condition_name = timing_file.name.removesuffix(".1D")

    upsampled_condition_regressor_file = (
        timing_file.parent / "upsampled" / f"{condition_name}_desc-upsampled.1D"
    )
    upsampled_condition_regressor_file.parent.mkdir(parents=True, exist_ok=True)

    duration = (
        CONDITION_DURATIONS[task]
        if not condition_name.startswith("instruction")
        else CONDITION_DURATIONS[f"{condition_name}_{task}"]
    )

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} timing_tool.py "
        f"-timing {timing_file} "
        f"-tr {upsample_dt} "
        f"-stim_dur {duration} "
        f"-run_len {tr * n_volumes} "
        f"-timing_to_1D {upsampled_condition_regressor_file}"
    )

    LGR.info(f"Upsampling condition {condition_name} to {upsample_dt} s: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return upsampled_condition_regressor_file


def create_convolved_ppi_term(
    ppi_dir,
    deconvolved_seed_timeseries_file,
    upsampled_condition_regressor_file,
    afni_img_path,
    upsample_dt,
):
    # waver -GAM -peak 1 -TR ?  -input Inter_neuA.1D -numout #TRs > Inter_A.1D
    # PPI = ([neural signal * binary_condition_vector] * hrf)(t)
    neural_interaction_file = (
        deconvolved_seed_timeseries_file.parent
        / upsampled_condition_regressor_file.name.replace(
            "_desc-upsampled", "_desc-neural_interaction"
        )
    )
    ppi_regressor_file = ppi_dir / upsampled_condition_regressor_file.name.replace(
        "_desc-upsampled", "_desc-PPI_upsampled"
    )

    numout = np.loadtxt(deconvolved_seed_timeseries_file).size

    # Create the interaction, which simply zeroes the parts when the condition is not active
    # Then reconvolve the interaction term to get the estimated HRF, ensure no extended tail due to convolution
    # So regressor can be properly downsampled
    cmd = (
        f'apptainer exec -B /projects:/projects {afni_img_path} bash -c "1deval '
        f"-a {deconvolved_seed_timeseries_file}\\' -b {upsampled_condition_regressor_file} "
        f"-expr 'a*b' > {neural_interaction_file} && "
        f"waver -GAM -peak 1 -TR {upsample_dt} "
        f'-input {neural_interaction_file} -numout {numout} > {ppi_regressor_file}"'
    )

    LGR.info(f"Reconvolving upsampled PPI regressor: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return ppi_regressor_file


def main(
    bids_dir,
    scratch_dir,
    afni_img_path,
    dst_dir,
    deriv_dir,
    seed_mask_path,
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
    upsample_dt,
    faltung_penalty_syntax,
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

        subject_analysis_dir = (
            Path(dst_dir) / f"sub-{subject}" / f"ses-{session}" / "func" / task
        )
        subject_analysis_dir.mkdir(parents=True, exist_ok=True)

        subject_scratch_dir = (
            Path(scratch_dir) / f"sub-{subject}" / f"ses-{session}" / "func" / task
        )
        subject_scratch_dir.mkdir(parents=True, exist_ok=True)

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
            subject_analysis_dir, subject, session, task, space, censor_mask
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
            subject_analysis_dir,
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

        # Note, only leaving task black because subject_analysis_dir includes the task name
        timing_dir = create_timing_files(subject_analysis_dir, event_file, task="")

        percent_change_nifti_file = percent_signal_change(
            subject_analysis_dir, afni_img_path, nifti_file, mask_file, censor_file
        )

        # gPPI preparation
        seed_mask_path = Path(seed_mask_path)

        tr = get_tr(nifti_file)
        n_volumes = get_n_volumes(nifti_file)

        ppi_dir = timing_dir / "ppi"
        ppi_dir.mkdir(parents=True, exist_ok=True)

        seed_timeseries_file = extract_seed_timeseries(
            subject_analysis_dir,
            subject_scratch_dir,
            percent_change_nifti_file,
            seed_mask_path,
            afni_img_path,
        )
        denoised_seed_timeseries_file = denoise_seed_timeseries(
            subject_analysis_dir,
            subject,
            session,
            task,
            space,
            seed_timeseries_file,
            censor_file,
            afni_img_path,
            confounds_df,
        )
        upsampled_seed_timeseries_file = resample_data(
            denoised_seed_timeseries_file,
            tr,
            afni_img_path,
            upsample_dt,
            method="upsample",
        )
        deconvolved_seed_timeseries_file = deconvolve_seed_timeseries(
            upsampled_seed_timeseries_file,
            upsample_dt,
            faltung_penalty_syntax,
            afni_img_path,
        )

        first_level_gltsym_codes = get_first_level_gltsym_codes(
            task, analysis_type="glm", caller="gPPI"
        )
        condition_filenames = [
            timing_dir / f"{condition}.1D"
            for condition in get_beta_names(first_level_gltsym_codes)
            if "_vs_" not in condition
        ]
        condition_filenames = get_instruction_name(
            timing_dir, task, condition_filenames
        )
        for condition_filename in condition_filenames:
            if is_timing_file_empty(condition_filename):
                continue

            upsampled_condition_regressor_file = upsample_condition_regressor(
                condition_filename, task, tr, n_volumes, upsample_dt, afni_img_path
            )
            ppi_regressor_file = create_convolved_ppi_term(
                ppi_dir,
                deconvolved_seed_timeseries_file,
                upsampled_condition_regressor_file,
                afni_img_path,
                upsample_dt,
            )
            resample_data(
                ppi_regressor_file, tr, afni_img_path, upsample_dt, method="downsample"
            )

        smoothed_nifti_file = perform_spatial_smoothing(
            subject_analysis_dir.parent,
            afni_img_path,
            percent_change_nifti_file,
            mask_file,
            fwhm,
        )

        deconvolve_cmd = get_task_deconvolve_cmd(
            task, timing_dir, nuisance_regressors_file, seed_timeseries_file, ppi_dir
        )
        design_matrix_file = create_design_matrix(
            subject_analysis_dir.parent,
            afni_img_path,
            smoothed_nifti_file,
            mask_file,
            censor_file,
            deconvolve_cmd,
            cosine_regressor_names,
        )

        stats_file_relm = perform_first_level(
            subject_analysis_dir.parent,
            afni_img_path,
            design_matrix_file,
            smoothed_nifti_file,
            mask_file,
        )

        betas_dir = stats_file_relm.parent / "betas"
        betas_dir.mkdir(parents=True, exist_ok=True)

        create_beta_files(
            stats_file_relm, betas_dir, afni_img_path, task, analysis_type="gPPI"
        )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
