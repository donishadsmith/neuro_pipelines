"""
gPPI workflow

Papers:
1) https://pmc.ncbi.nlm.nih.gov/articles/PMC4632075/
2) https://pmc.ncbi.nlm.nih.gov/articles/PMC3376181/

Forums:
1) https://discuss.afni.nimh.nih.gov/t/gppi-analysis-and-upsampling/172
2) https://web.archive.org/web/20241103095511/https://afni.nimh.nih.gov/CD-CorrAna
   (archived website)
3) https://discuss.afni.nimh.nih.gov/t/how-to-do-gppi-to-event-related-fmri/457/4

1) Collect confounds and create censor file
2) PSC scaling of NIfTI image, compute mean for censored files
3) Resample mask to NIfTI (if needed) then extract timeseries
4) Tranpose the seed timeseries to a column vector
5) Denoise seed timeseries. Do same denoising with seed and image. Ensure regular OLS is used just
to orthogonalize to the nuisance regressors for the seed, no prewhitening should be done to prevent
temporal autocorrelation in the residuals. Note that smoothing is not done prior
to extracting the seed, the timeseries is already averaged which helps with spatial noise
reduction. More importantly, smoothing blur signal outside of the voxels of interest into
the will result in your seed timeseries containing signal from voxels outside of your mask

For each condition in task (6-10):
6) Upsample seed timeseries (https://www.nature.com/articles/s42003-024-07088-3) and task regressor to 0.1
   (TR_orig/ TR_sub is equal to number of points added between each TR or
   the duration / TR_sub is equal to the number of points added after each onset
   time).
   Resources:
        1) https://discuss.afni.nimh.nih.gov/t/gppi-analysis-and-upsampling/172
        2) https://www.nature.com/articles/s42003-024-07088-3
7) The task regressor should then be mean centered so that the subsequent interaction term
   is not highly correlated with the main effect of the seed timeseries and result in
   spurious results that attribute correlation with the seed timeseries to the interaction
   term. Great paper about this:
   https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.989/133601/Common-pitfalls-during-model-specification-in
8) Deconvolve seed timeseries to get the neural signal that will later
   interact with the task regressor and this interaction will be convolved.
9) Create PPI term PPI = ([neural signal * binary_condition_vector] * hrf)(t).
   Use GAM for event-related tasks, and a simulated BLOCK function for block-design tasks.
10) Downsample the PPI term back down to the true TR grid

After:
11) For NIfTI image, smooth, then use 3ddeconvolve. Ensure to model everything
   from nuisance regressors, all main effect conditions (convolved), the
   denoised seed signal, and the PPI interaction terms (already convolved
   in previous step). Create contrasts of the interaction terms (+ means
   greater connectivity for A than B and - means reduced connectivity for
   A relative to B)
12) Use 3dremlfit to account for temporal autocorrelation
13) Extract PPI interaction contrasts betas for downstream analyses

# Interpretation:
- Positive beta coefficients for the PPI regressor means greater connectivity between
  the seed region and the brain region during a specific condition
- Negative beta coefficients for the PPI regressor means reduced connectivity between
  the seed region and the brain region during a specific condition


Denoising papers:
    - https://pmc.ncbi.nlm.nih.gov/articles/PMC7978116/

- Aggressive denoising strategies can also remove task-signal, strategy should depend on
clinical population, type of analysis being done (activation vs connectivity, where in some cases
connectivity may require more aggressive denoising to ensure that correlation is not due to
shared nuisance variance), characteristics of data (numerous high-motion participants or
mostly low-motion participants), and whether strategies such as strict scrubbing (FD < 0.2) will
remove a significant amount of frames resulting in either suboptimal estimated beta coefficients
or too little retainerd participants. There is no optimal denoising strategy for all datasets.
"""

import argparse, json, subprocess, sys
from pathlib import Path

import nibabel as nib, numpy as np

import bids, numpy as np, pandas as pd

from bidsaid._helpers import iterable_to_str
from bidsaid.logging import setup_logger
from bidsaid.metadata import get_tr, get_n_volumes
from bidsaid.qc import (
    compute_n_dummy_scans,
    create_censor_mask,
)

from _denoising import (
    get_acompcor_component_names,
    get_cosine_regressors,
    get_motion_regressors,
    percent_signal_change,
    perform_spatial_smoothing,
)
from _gen_afni_files import (
    create_censor_file,
    create_binary_condition,
    create_timing_files,
    create_nuisance_regressor_file,
    is_timing_file_empty,
)
from _argparse_typing import n_dummy_type, boolean_flags
from _models import create_design_matrix, perform_first_level
from _report import HTMLReport
from _utils import (
    CONDITION_DURATIONS,
    VALID_TASK_NAMES,
    create_beta_files,
    embed_image,
    get_beta_names,
    get_coordinate_from_filename,
    get_first_level_gltsym_codes,
    plot_signal,
    resample_seed_img,
    skip_denoising,
)

LGR = setup_logger(__name__)

EVENT_RELATED_TASKS = ["flanker", "simplegng", "complexgng"]


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
        "--cohort",
        dest="cohort",
        required=True,
        choices=["adults", "kids"],
        help="The cohort to analyze.",
    )
    parser.add_argument(
        "--space",
        dest="space",
        required=True,
        help="Template space (i.e. 'MNIPediatricAsym_cohort-1_res-2')",
    )
    parser.add_argument(
        "--subject",
        dest="subject",
        required=True,
        help="Subject ID without the 'sub-' entity.",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--filter_correct_trials",
        dest="filter_correct_trials",
        required=False,
        default=False,
        type=boolean_flags,
        help="Filter correct trials for event-related tasks.",
    )
    parser.add_argument(
        "--n_motion_parameters",
        dest="n_motion_parameters",
        default=6,
        type=int,
        choices=[6, 12, 18, 24],
        required=False,
        help=(
            "Number of motion parameters to use: 6 (base trans + rot), "
            "12 (base + derivatives), 18 (base + derivatives + power), "
            "24 (base + derivatives + power + derivative power). "
        ),
    )
    parser.add_argument(
        "--fd_threshold",
        dest="fd_threshold",
        default=0.5,
        type=float,
        required=False,
        help="Framewise displacement threshold.",
    )
    parser.add_argument(
        "--exclusion_criteria",
        dest="exclusion_criteria",
        default=0.30,
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
        default="separate",
        choices=["combined", "separate", "none"],
        type=str,
        required=False,
        help="Whether to use 'combined' aCompCor, 'separate' components, or 'none'.",
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
        help="Time resolution to upsample seed timeseries (and condition times) to prior to deconvolution.",
    )
    parser.add_argument(
        "--pad_seconds",
        dest="pad_seconds",
        default=10.0,
        type=float,
        required=False,
        help=(
            "Time in seconds to determine the padding to add to both ends (pad_seconds/upsample_dt) "
            "to minimize boundary spikes prior to deconvolution. The padding is dropped immediatelly afterwards "
            "so the final deconvolved timeseries includes no padding."
        ),
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
    parser.add_argument(
        "--exclude_nifti_files",
        dest="exclude_nifti_files",
        default=None,
        required=False,
        help=(
            "Prefixes of the filename of the NIfTI images to exclude. "
            "Can list the fill name of the file (no parent directories) to exlude that specific file "
            "or can include the prefix (i.e., 'sub-101_task-nback_ses-01_space-MNI' or 'sub-101') to exclude all files starting "
            "with that prefix. Should contain a single column named 'nifti_prefix_filename' "
        ),
    )

    return parser


def extract_seed_timeseries(
    subject_dir,
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

    seed_timeseries_file = subject_dir / "seed" / f"{seed_name}_desc-timeseries.1D"
    seed_timeseries_file.parent.mkdir(parents=True, exist_ok=True)

    seed_img = resample_seed_img(nib.load(seed_mask_path), nib.load(subject_nifti_file))

    resampled_seed_file = subject_dir / f"resampled_{seed_mask_path.name}"
    nib.save(seed_img, resampled_seed_file)

    # Note: output is a column vector
    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dmaskave "
        f"-mask {resampled_seed_file} "
        f"-q {subject_nifti_file} > {seed_timeseries_file}"
    )

    LGR.info(f"Extracting seed: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    resampled_seed_file.unlink()

    return seed_timeseries_file


def denoise_seed_timeseries(
    seed_timeseries_file,
    nuisance_regressors_file,
    censor_file,
    afni_img_path,
    cosine_regressor_names,
):

    polort = 0 if cosine_regressor_names else "A"
    LGR.info(f"Using polort {polort} for 3dTproject.")

    denoised_seed_timeseries_file = (
        seed_timeseries_file.parent
        / seed_timeseries_file.name.replace("_desc-timeseries", "_desc-denoised")
    )

    # Note: Some Afni functions only accept rows and require \', using \\' to
    # make the backslash literal
    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dTproject "
        f"-input {seed_timeseries_file}\\' "
        f"-ort {nuisance_regressors_file} "
        f"-polort {polort} "
        f"-censor {censor_file} "
        "-cenmode ZERO "
        f"-prefix {denoised_seed_timeseries_file}"
    )

    LGR.info(
        f"Denoising seed (same nuisance regressors used for seed and BOLD/NIfTI image): {cmd}"
    )
    subprocess.run(cmd, shell=True, check=True)

    return denoised_seed_timeseries_file


def get_task_deconvolve_kids_cmd(
    task, timing_dir, nuisance_regressors_file, seed_timeseries_file, ppi_dir
):
    seed_name = Path(seed_timeseries_file).name.split("_desc")[0]

    if task == "nback":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 9 ",
            "args": f"-stim_file 1 {seed_timeseries_file} -stim_label 1 {seed_name} "
            f"-stim_times 2 {timing_dir / 'instruction.1D'} 'BLOCK(2, 1)' -stim_label 2 instruction "
            f"-stim_times 3 {timing_dir / 'center.1D'} 'BLOCK(32, 1)' -stim_label 3 center "
            f"-stim_times 4 {timing_dir / '1-back.1D'} 'BLOCK(32, 1)' -stim_label 4 1-back "
            f"-stim_times 5 {timing_dir / '2-back.1D'} 'BLOCK(32, 1)' -stim_label 5 2-back "
            f"-stim_file 6 {ppi_dir / 'PPI_instruction.1D'} -stim_label 6 PPI_instruction "
            f"-stim_file 7 {ppi_dir / 'PPI_center.1D'} -stim_label 7 PPI_center "
            f"-stim_file 8 {ppi_dir / 'PPI_1-back.1D'} -stim_label 8 PPI_1-back "
            f"-stim_file 9 {ppi_dir / 'PPI_2-back.1D'} -stim_label 9 PPI_2-back "
            f"-ortvec {nuisance_regressors_file} Nuisance "
            "-gltsym 'SYM: +1*PPI_1-back -1*PPI_center' -glt_label 1 PPI_1-back_vs_PPI_center "
            "-gltsym 'SYM: +1*PPI_2-back -1*PPI_center' -glt_label 2 PPI_2-back_vs_PPI_center "
            "-gltsym 'SYM: +1*PPI_2-back -1*PPI_1-back' -glt_label 3 PPI_2-back_vs_PPI_1-back ",
        }
    elif task == "mtle":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 5 ",
            "args": f"-stim_file 1 {seed_timeseries_file} -stim_label 1 {seed_name} "
            f"-stim_times 2 {timing_dir / 'instruction.1D'} 'BLOCK(2, 1)' -stim_label 2 instruction "
            f"-stim_times 3 {timing_dir / 'neutral_encoding.1D'} 'BLOCK(18, 1)' -stim_label 3 neutral_encoding "
            f"-stim_file 4 {ppi_dir / 'PPI_instruction.1D'} -stim_label 4 PPI_instruction "
            f"-stim_file 5 {ppi_dir / 'PPI_neutral_encoding.1D'} -stim_label 5 PPI_neutral_encoding "
            f"-ortvec {nuisance_regressors_file} Nuisance ",
        }
    elif task == "mtlr":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 5 ",
            "args": f"-stim_file 1 {seed_timeseries_file} -stim_label 1 {seed_name} "
            f"-stim_times 2 {timing_dir / 'instruction.1D'} 'BLOCK(2, 1)' -stim_label 2 instruction "
            f"-stim_times 3 {timing_dir / 'neutral_retrieval.1D'} 'BLOCK(18, 1)' -stim_label 3 neutral_retrieval "
            f"-stim_file 4 {ppi_dir / 'PPI_instruction.1D'} -stim_label 4 PPI_instruction "
            f"-stim_file 5 {ppi_dir / 'PPI_neutral_retrieval.1D'} -stim_label 5 PPI_neutral_retrieval "
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
        deconvolve_cmd = create_dynamic_deconvolve_gPPI_cmd(
            timing_dir, nuisance_regressors_file, seed_timeseries_file, ppi_dir, task
        )

    return deconvolve_cmd


def get_task_deconvolve_adults_cmd(
    task, timing_dir, nuisance_regressors_file, seed_timeseries_file, ppi_dir
):
    seed_name = Path(seed_timeseries_file).name.split("_desc")[0]

    if task == "nback":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 5 ",
            "args": f"-stim_file 1 {seed_timeseries_file} -stim_label 1 {seed_name} "
            f"-stim_times 2 {timing_dir / '0-back.1D'} 'BLOCK(30, 1)' -stim_label 2 0-back "
            f"-stim_times 3 {timing_dir / '2-back.1D'} 'BLOCK(30, 1)' -stim_label 3 2-back "
            f"-stim_file 4 {ppi_dir / 'PPI_0-back.1D'} -stim_label 4 PPI_0-back "
            f"-stim_file 5 {ppi_dir / 'PPI_2-back.1D'} -stim_label 5 PPI_2-back "
            f"-ortvec {nuisance_regressors_file} Nuisance "
            "-gltsym 'SYM: +1*PPI_2-back -1*PPI_0-back' -glt_label 1 PPI_2-back_vs_PPI_0-back ",
        }
    elif task == "mtle":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 7 ",
            "args": f"-stim_file 1 {seed_timeseries_file} -stim_label 1 {seed_name} "
            f"-stim_times 2 {timing_dir / 'instruction.1D'} 'BLOCK(2, 1)' -stim_label 2 instruction "
            f"-stim_times 3 {timing_dir / 'neutral_encoding.1D'} 'BLOCK(18, 1)' -stim_label 3 neutral_encoding "
            f"-stim_times 4 {timing_dir / 'aversive_encoding.1D'} 'BLOCK(18, 1)' -stim_label 4 aversive_encoding "
            f"-stim_file 5 {ppi_dir / 'PPI_instruction.1D'} -stim_label 5 PPI_instruction "
            f"-stim_file 6 {ppi_dir / 'PPI_neutral_encoding.1D'} -stim_label 6 PPI_neutral_encoding"
            f"-stim_file 7 {ppi_dir / 'PPI_aversive_encoding.1D'} -stim_label 7 PPI_aversive_encoding "
            f"-ortvec {nuisance_regressors_file} Nuisance "
            "-gltsym 'SYM: +1*PPI_aversive_encoding -1*PPI_neutral_encoding' -glt_label 1 PPI_aversive_encoding_vs_PPI_neutral_encoding ",
        }
    elif task == "mtlr":
        deconvolve_cmd = {
            "num_stimts": "-num_stimts 7 ",
            "args": f"-stim_file 1 {seed_timeseries_file} -stim_label 1 {seed_name} "
            f"-stim_times 2 {timing_dir / 'instruction.1D'} 'BLOCK(2, 1)' -stim_label 2 instruction "
            f"-stim_times 3 {timing_dir / 'neutral_retrieval.1D'} 'BLOCK(18, 1)' -stim_label 3 neutral_retrieval "
            f"-stim_times 4 {timing_dir / 'aversive_retrieval.1D'} 'BLOCK(18, 1)' -stim_label 4 aversive_retrieval "
            f"-stim_file 5 {ppi_dir / 'PPI_instruction.1D'} -stim_label 5 PPI_instruction "
            f"-stim_file 6 {ppi_dir / 'PPI_neutral_retrieval.1D'} -stim_label 6 PPI_neutral_retrieval"
            f"-stim_file 7 {ppi_dir / 'PPI_aversive_retrieval.1D'} -stim_label 7 PPI_aversive_retrieval "
            f"-ortvec {nuisance_regressors_file} Nuisance "
            "-gltsym 'SYM: +1*PPI_aversive_retrieval -1*PPI_neutral_retrieval' -glt_label 1 PPI_aversive_retrieval_vs_PPI_neutral_retrieval ",
        }
    else:
        deconvolve_cmd = create_dynamic_deconvolve_gPPI_cmd(
            timing_dir, nuisance_regressors_file, seed_timeseries_file, ppi_dir, task
        )

    return deconvolve_cmd


def create_dynamic_deconvolve_gPPI_cmd(
    timing_dir, nuisance_regressors_file, seed_timeseries_file, ppi_dir, task
):
    # Dynamically create the flanker and nogo contrasts to avoid including contrasts that
    # have no data

    deconvolve_cmd = {
        "num_stimts": "-num_stimts {num_labels} ",
        "args": "{stims} -ortvec {nuisance_regressors_file} Nuisance {gltsyms}",
    }

    if task == "flanker":
        labels_dict = {
            "stims": (
                "-stim_times {label} {timing_file} 'GAM' -stim_label {label} congruent ",
                "-stim_times {label} {timing_file} 'GAM' -stim_label {label} incongruent ",
                "-stim_times {label} {timing_file} 'GAM' -stim_label {label} nogo ",
                "-stim_times {label} {timing_file} 'GAM' -stim_label {label} neutral ",
                "-stim_times {label} {timing_file} 'GAM' -stim_label {label} errors ",
                "-stim_file {label} {ppi_file} -stim_label {label} PPI_congruent ",
                "-stim_file {label} {ppi_file} -stim_label {label} PPI_incongruent ",
                "-stim_file {label} {ppi_file} -stim_label {label} PPI_nogo ",
                "-stim_file {label} {ppi_file} -stim_label {label} PPI_neutral ",
                "-stim_file {label} {ppi_file} -stim_label {label} PPI_errors ",
            ),
            "gltsyms": (
                "-gltsym 'SYM: +1*PPI_incongruent -1*PPI_congruent' -glt_label {label} PPI_incongruent_vs_PPI_congruent ",
                "-gltsym 'SYM: +1*PPI_nogo -1*PPI_neutral' -glt_label {label} PPI_nogo_vs_PPI_neutral ",
            ),
        }

        files = [
            "congruent.1D",
            "incongruent.1D",
            "nogo.1D",
            "neutral.1D",
            "errors.1D",
        ]
    else:
        labels_dict = {
            "stims": (
                "-stim_times {label} {timing_file} 'GAM' -stim_label {label} go ",
                "-stim_times {label} {timing_file} 'GAM' -stim_label {label} nogo ",
                "-stim_times {label} {timing_file} 'GAM' -stim_label {label} errors ",
                "-stim_file {label}  {ppi_file} -stim_label {label} PPI_go ",
                "-stim_file {label} {ppi_file} -stim_label {label} PPI_nogo ",
                "-stim_file {label} {ppi_file} -stim_label {label} PPI_errors ",
            ),
            "gltsyms": (
                "-gltsym 'SYM: +1*PPI_nogo -1*PPI_go' -glt_label {label} PPI_nogo_vs_PPI_go ",
            ),
        }

        files = ["go.1D", "nogo.1D", "errors.1D"]

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
                label=label,
                timing_file=timing_dir / f"{regressor}.1D",
            )

    stims = stims.rstrip()

    # Length of the stims
    deconvolve_cmd["num_stimts"] = deconvolve_cmd["num_stimts"].format(num_labels=label)

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


def get_instruction_name(timing_dir, cohort, task, condition_filenames):
    if task not in ["nback", "mtle", "mtlr"] and not (
        task == "nback" and cohort == "adult"
    ):
        return condition_filenames
    else:
        return condition_filenames + [timing_dir / f"instruction.1D"]


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
    upsampled_seed_timeseries_file,
    upsample_dt,
    pad_seconds,
    faltung_penalty_syntax,
    afni_img_path,
    task,
):
    gamma_file_name = upsampled_seed_timeseries_file.parent / "GammaHR.1D"
    deconvolved_seed_timeseries_file = (
        upsampled_seed_timeseries_file.parent
        / upsampled_seed_timeseries_file.name.replace(
            "_desc-upsampled", "_desc-deconvolved"
        )
    )

    padded_deconvolved_seed_timeseries_file = (
        deconvolved_seed_timeseries_file.parent
        / deconvolved_seed_timeseries_file.name.replace(
            "_desc-deconvolved", "_desc-deconvolved_padded"
        )
    )

    # Use some padding for smooth ramp up at ends
    pad_length = int(pad_seconds / upsample_dt)
    padded_upsampled_seed_timeseries_file = (
        upsampled_seed_timeseries_file.parent
        / upsampled_seed_timeseries_file.name.replace(
            "_desc-upsampled", "_desc-upsampled_padded"
        )
    )
    padded_arr = np.pad(
        np.loadtxt(upsampled_seed_timeseries_file), pad_width=pad_length, mode="reflect"
    )
    np.savetxt(
        padded_upsampled_seed_timeseries_file, padded_arr.reshape(-1, 1), fmt="%f"
    )

    # https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDeconvolve.html
    # https://doi.org/10.1002/hbm.26047
    # Creating 30 second hrf
    hrf_model = "GAM" if task in EVENT_RELATED_TASKS else f"BLOCK({upsample_dt},1)"

    hrf_cmd = (
        f"3dDeconvolve -nodata {int(30 / upsample_dt)} {upsample_dt} -polort -1 "
        f"-num_stimts 1 -stim_times 1 '1D: 0' '{hrf_model}' "
        f"-x1D {gamma_file_name}_tmp -x1D_stop -quiet && "
        f"1dcat {gamma_file_name}_tmp > {gamma_file_name}"
    )

    # Perform deconvolution to estimate the neural response given the upsampled seed timeseries
    # and an hrf response function, while also adding a penalty for better/smoother estimation
    cmd = (
        f'apptainer exec -B /projects:/projects {afni_img_path} bash -c "{hrf_cmd} && '
        f"3dTfitter -RHS {padded_upsampled_seed_timeseries_file} "
        f'-FALTUNG {gamma_file_name} {padded_deconvolved_seed_timeseries_file} {faltung_penalty_syntax}"'
    )

    LGR.info(f"Deconvolving upsampled seed timeseries: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    deconvolved_arr = np.loadtxt(padded_deconvolved_seed_timeseries_file)[
        pad_length:-pad_length
    ]
    np.savetxt(
        deconvolved_seed_timeseries_file, deconvolved_arr.reshape(-1, 1), fmt="%f"
    )

    padded_upsampled_seed_timeseries_file.unlink()
    padded_deconvolved_seed_timeseries_file.unlink()
    Path(f"{gamma_file_name}_tmp").unlink()

    return deconvolved_seed_timeseries_file


def upsample_condition_regressor(
    timing_file, cohort, task, tr, n_volumes, upsample_dt, afni_img_path
):
    condition_name = timing_file.name.removesuffix(".1D")

    upsampled_condition_regressor_file = (
        timing_file.parent / "upsampled" / f"{condition_name}_desc-upsampled.1D"
    )
    upsampled_condition_regressor_file.parent.mkdir(parents=True, exist_ok=True)

    duration = (
        CONDITION_DURATIONS[cohort][task]
        if not condition_name.startswith("instruction")
        else CONDITION_DURATIONS[cohort][f"{condition_name}_{task}"]
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

    # Now mean center the task regressor
    condition_vector = np.loadtxt(upsampled_condition_regressor_file)
    condition_vector -= condition_vector.mean()
    np.savetxt(
        upsampled_condition_regressor_file, condition_vector.reshape(-1, 1), fmt="%f"
    )

    return upsampled_condition_regressor_file


def create_convolved_ppi_term(
    ppi_dir,
    deconvolved_seed_timeseries_file,
    upsampled_condition_regressor_file,
    afni_img_path,
    upsample_dt,
):
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
    gamma_file_name = deconvolved_seed_timeseries_file.parent / "GammaHR.1D"

    # PPI = ([neural signal * binary_condition_vector] * hrf)(t)
    convolution_cmd = f"waver -FILE {upsample_dt} {gamma_file_name} -peak 1 -input {neural_interaction_file} -numout {numout} > {ppi_regressor_file}"

    # Create the interaction, which simply zeroes the parts when the condition is not active
    # Then reconvolve the interaction term to get the estimated HRF, ensure no extended tail due to convolution
    # So regressor can be properly downsampled
    cmd = (
        f'apptainer exec -B /projects:/projects {afni_img_path} bash -c "1deval '
        f"-a {deconvolved_seed_timeseries_file} -b {upsampled_condition_regressor_file} "
        f"-expr 'a*b' > {neural_interaction_file} && "
        f'{convolution_cmd}"'
    )

    LGR.info(f"Reconvolving upsampled PPI regressor: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return ppi_regressor_file


def main(
    bids_dir,
    afni_img_path,
    dst_dir,
    deriv_dir,
    seed_mask_path,
    cohort,
    subject,
    space,
    task,
    filter_correct_trials,
    n_motion_parameters,
    fd_threshold,
    exclusion_criteria,
    n_dummy_scans,
    n_acompcor,
    acompcor_strategy,
    fwhm,
    upsample_dt,
    pad_seconds,
    faltung_penalty_syntax,
    exclude_nifti_files,
):
    report_dir = Path(dst_dir) / "reports" / "first_level"
    report_dir.mkdir(parents=True, exist_ok=True)

    if task not in VALID_TASK_NAMES[cohort]:
        LGR.warning(
            f"The task must be one of the following: {iterable_to_str(VALID_TASK_NAMES[cohort])}"
        )
        sys.exit(status=1)

    layout = bids.BIDSLayout(bids_dir, derivatives=deriv_dir or True)

    sessions = layout.get(
        subject=subject, task=task, target="session", return_type="id"
    )
    if not sessions:
        session = "NaN"
        report = HTMLReport(subject, session, task, analysis_type="gPPI")
        report_path = (
            report_dir / f"sub-{subject}_ses-NaN_task-{task}_desc-gPPI_report.html"
        )
        msg = f"No sessions for {subject} for {task}."
        LGR.warning(msg)

        report.mark_excluded(msg)
        report.create_report(report_path, "first_level.html")

        sys.exit(status=1)

    for session in sessions:
        report = HTMLReport(subject, session, task, analysis_type="gPPI")
        report_path = (
            report_dir
            / f"sub-{subject}_ses-{session}_task-{task}_desc-gPPI_report.html"
        )

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
            msg = f"No confound files TSV found for session: {session}"
            LGR.info(msg)

            report.mark_excluded(msg)
            report.create_report(report_path, "first_level.html")
            continue
        else:
            confounds_tsv_file = confounds_tsv_files[0]

        if acompcor_strategy != "none":
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
                msg = f"No confound files JSON found for session: {session}"
                LGR.info(msg)

                report.mark_excluded(msg)
                report.create_report(report_path, "first_level.html")
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
            msg = f"No event files found for session: {session}"
            LGR.info(msg)

            report.mark_excluded(msg)
            report.create_report(report_path, "first_level.html")
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
            msg = f"No mask files found for session: {session}"
            LGR.info(msg)

            report.mark_excluded(msg)
            report.create_report(report_path, "first_level.html")
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
            msg = f"No nifti files found for session: {session}"
            LGR.info(msg)

            report.mark_excluded(msg)
            report.create_report(report_path, "first_level.html")
            continue
        else:
            nifti_file = [
                file for file in nifti_files if space in str(Path(file).name)
            ][0]
            LGR.info(f"Using the following mask file: {nifti_file}")

        subject_dir = (
            Path(dst_dir) / f"sub-{subject}" / f"ses-{session}" / "func" / task
        )

        if skip_denoising(nifti_file, exclude_nifti_files):
            LGR.info(
                "Denoising of the following file will be skipped due to the prefix being found in "
                f"`exclude_nifti_files` ({exclude_nifti_files}): {nifti_file}"
            )

            report.mark_excluded(
                f"Skipped due to prefix being found in {exclude_nifti_files} "
            )
            report.create_report(report_path, "first_level.html")
            continue

        subject_dir.mkdir(parents=True, exist_ok=True)

        confounds_df = pd.read_csv(confounds_tsv_file, sep="\t").fillna(0)

        dummy_method = (
            "user-specified"
            if n_dummy_scans != "auto"
            else "(number of 'non_steady_state_outlier_XX' columns in fMRIPrep confounds TSV)"
        )
        if n_dummy_scans == "auto":
            n_dummy_scans = compute_n_dummy_scans(confounds_df)
            LGR.info(f"There are {n_dummy_scans} non-steady state scans.")

        censor_mask = create_censor_mask(
            confounds_df,
            column_name="framewise_displacement",
            n_dummy_scans=n_dummy_scans,
            threshold=fd_threshold,
        )
        censor_mask = censor_mask.astype(np.int8)

        kept = censor_mask[n_dummy_scans:]
        n_censored = np.sum(kept == 0)
        percent_censored = n_censored / kept.size
        LGR.warning(
            f"For SUBJECT: {subject}, SESSION: {session}, TASK: {task}, "
            f"proportion of steady state volumes removed at an fd threshold > {fd_threshold} mm: "
            f" {percent_censored}"
        )

        report.add_context(
            fd_threshold=fd_threshold,
            exclusion_criteria=exclusion_criteria,
            n_censored_volumes=int(n_censored),
            n_total_volumes=int(kept.size),
            percent_censored=float(percent_censored),
            dummy_method=dummy_method,
            n_dummy_scans=n_dummy_scans,
        )

        if percent_censored > exclusion_criteria:
            LGR.warning(
                f"For SUBJECT: {subject}, SESSION: {session}, TASK: {task}, "
                "run excluded because the percent censored is greater than the "
                f"exclusion criteria: {exclusion_criteria}"
            )

            report.mark_excluded(
                f"Proportion of flagged volumes ({percent_censored:.1%}) "
                f"exceeded threshold ({exclusion_criteria:.0%})."
            )
            report.create_report(report_path, "first_level.html")
            continue

        censor_file = create_censor_file(
            subject_dir, subject, session, task, space, censor_mask
        )

        cosine_regressors, cosine_regressor_names = get_cosine_regressors(confounds_df)

        motion_regressors, motion_regressor_names = get_motion_regressors(
            confounds_df, n_motion_parameters
        )

        if acompcor_strategy == "none":
            acompcor_regressors, acompcor_regressor_names = None, None
        else:
            with open(confounds_json_file, "r") as f:
                confounds_meta = json.load(f)

            acompcor_regressor_names = get_acompcor_component_names(
                confounds_meta, n_acompcor, acompcor_strategy
            )
            acompcor_regressors = confounds_df[acompcor_regressor_names].to_numpy(
                copy=True
            )

        report.add_context(
            n_motion_parameters=n_motion_parameters,
            motion_regressor_names=motion_regressor_names,
            acompcor_strategy=acompcor_strategy,
            n_acompcor=n_acompcor,
            acompcor_component_names=acompcor_regressor_names or [],
            cosine_parameter_names=cosine_regressor_names,
            fwhm=fwhm,
            filter_correct_trials=filter_correct_trials,
        )

        regressor_names_nested_list = filter(
            None,
            [
                cosine_regressor_names,
                motion_regressor_names,
                acompcor_regressor_names,
            ],
        )
        regressor_names = [
            regressor
            for regressor_list in regressor_names_nested_list
            for regressor in regressor_list
        ]
        nuisance_regressors_file, report_info = create_nuisance_regressor_file(
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
            analysis_type="gPPI",
        )

        dropped_regressors = (
            report_info["collinear_regressor_names"]
            + report_info["constant_column_names"]
        )
        if dropped_regressors:
            report.add_context(dropped_regressors=dropped_regressors)

        timing_dir = create_timing_files(
            subject_dir,
            event_file,
            task=task,
            filter_correct_trials=filter_correct_trials,
            append_task_name=False,
        )

        tr = get_tr(nifti_file)
        n_volumes = get_n_volumes(nifti_file)

        condition_filenames_dict = create_binary_condition(
            afni_img_path, timing_dir, cohort, task, tr, n_volumes, censor_file
        )

        diagnostic_condition_plots = []
        for cond_name, cond_vector_files in condition_filenames_dict.items():
            noncensored_condition_plotname = plot_signal(
                cond_vector_files["noncensored_binary_vector"],
                tr,
                plot_title=f"{cond_name} No Motion Censoring",
            )

            censored_condition_plotname = plot_signal(
                cond_vector_files["censored_binary_vector"],
                tr,
                plot_title=f"{cond_name} Censored (FD = {fd_threshold})",
            )

            diagnostic_condition_plots.append(
                {
                    "name": cond_name,
                    "noncensored_condition_plot": embed_image(
                        noncensored_condition_plotname
                    ),
                    "censored_condition_plot": embed_image(censored_condition_plotname),
                }
            )

        report.add_context(
            diagnostic_condition_plots=diagnostic_condition_plots,
        )

        timing_conditions = []
        for tf in sorted(timing_dir.glob("*.1D")):
            data = np.loadtxt(tf, delimiter=" ")
            timing_conditions.append(
                {
                    "name": tf.stem,
                    "n_events": int(data.size) if data.size > 0 else 0,
                }
            )

        report.add_context(
            timing_conditions=timing_conditions,
            event_type=(
                "blocks"
                if task not in ["flanker", "simplegng", "complexgng"]
                else "events"
            ),
        )

        percent_change_nifti_file = percent_signal_change(
            subject_dir, afni_img_path, nifti_file, mask_file, censor_file
        )

        # gPPI preparation
        seed_mask_path = Path(seed_mask_path)

        hrf_model_type = (
            "GAM" if task in EVENT_RELATED_TASKS else f"BLOCK({upsample_dt}, 1)"
        )
        hrf_model_desc = (
            "A standard Gamma (GAM) function was used to model the impulse response for this event-related task."
            if task in EVENT_RELATED_TASKS
            else f"A custom {upsample_dt}s duration BLOCK function was simulated via 3dDeconvolve to model the impulse response for this block-design task."
        )

        report.add_context(
            seed_mask_path=str(seed_mask_path),
            seed_coordinate=get_coordinate_from_filename(seed_mask_path),
            upsample_dt=upsample_dt,
            pad_seconds=pad_seconds,
            pad_length=int(pad_seconds / upsample_dt),
            faltung_penalty_syntax=faltung_penalty_syntax,
            tr=tr,
            hrf_model_type=hrf_model_type,
            hrf_model_desc=hrf_model_desc,
        )

        ppi_dir = timing_dir / "ppi"
        ppi_dir.mkdir(parents=True, exist_ok=True)

        seed_timeseries_file = extract_seed_timeseries(
            subject_dir,
            percent_change_nifti_file,
            seed_mask_path,
            afni_img_path,
        )
        plot_title = "Seed Timeseries"
        seed_timeseries_plot_filename = plot_signal(
            seed_timeseries_file, tr, plot_title
        )

        denoised_seed_timeseries_file = denoise_seed_timeseries(
            seed_timeseries_file,
            nuisance_regressors_file,
            censor_file,
            afni_img_path,
            cosine_regressor_names,
        )
        plot_title = "Denoised Seed Timeseries"
        denoised_seed_timeseries_plot_filename = plot_signal(
            denoised_seed_timeseries_file, tr, plot_title
        )

        upsampled_seed_timeseries_file = resample_data(
            denoised_seed_timeseries_file,
            tr,
            afni_img_path,
            upsample_dt,
            method="upsample",
        )
        plot_title = "Upsampled Seed Timeseries"
        upsampled_seed_timeseries_plot_filename = plot_signal(
            upsampled_seed_timeseries_file, tr, plot_title, upsample_dt
        )

        deconvolved_seed_timeseries_file = deconvolve_seed_timeseries(
            upsampled_seed_timeseries_file,
            upsample_dt,
            pad_seconds,
            faltung_penalty_syntax,
            afni_img_path,
            task,
        )
        plot_title = "Deconvolved Seed Timeseries"
        deconvolved_seed_timeseries_plot_filename = plot_signal(
            deconvolved_seed_timeseries_file, tr, plot_title, upsample_dt
        )

        report.add_context(
            seed_timeseries_plot=embed_image(seed_timeseries_plot_filename),
            denoised_seed_timeseries_plot=embed_image(
                denoised_seed_timeseries_plot_filename
            ),
            upsampled_seed_timeseries_plot=embed_image(
                upsampled_seed_timeseries_plot_filename
            ),
            deconvolved_seed_timeseries_plot=embed_image(
                deconvolved_seed_timeseries_plot_filename
            ),
        )

        first_level_gltsym_codes = get_first_level_gltsym_codes(
            cohort, task, analysis_type="glm", caller="gPPI"
        )
        condition_filenames = [
            timing_dir / f"{condition}.1D"
            for condition in get_beta_names(first_level_gltsym_codes)
            if "_vs_" not in condition
        ]
        condition_filenames = get_instruction_name(
            timing_dir, cohort, task, condition_filenames
        )

        condition_names = []
        condition_plots = []
        for condition_filename in condition_filenames:
            if is_timing_file_empty(condition_filename):
                continue

            cond_name = condition_filename.name.removesuffix(".1D")
            condition_names.append(cond_name)

            upsampled_condition_regressor_file = upsample_condition_regressor(
                condition_filename,
                cohort,
                task,
                tr,
                n_volumes,
                upsample_dt,
                afni_img_path,
            )
            plot_title = f"{cond_name.capitalize()} Upsampled Condition Regressor"
            upsampled_condition_regressor_plot_filename = plot_signal(
                upsampled_condition_regressor_file, tr, plot_title, upsample_dt
            )

            ppi_regressor_file = create_convolved_ppi_term(
                ppi_dir,
                deconvolved_seed_timeseries_file,
                upsampled_condition_regressor_file,
                afni_img_path,
                upsample_dt,
            )
            plot_title = f"{cond_name.capitalize()} Upsampled PPI Timeseries"
            upsampled_ppi_plot_filename = plot_signal(
                ppi_regressor_file, tr, plot_title, upsample_dt
            )

            downsampled_ppi_regressor_file = resample_data(
                ppi_regressor_file, tr, afni_img_path, upsample_dt, method="downsample"
            )
            plot_title = f"{cond_name.capitalize()} Downsampled PPI Timeseries"
            downsampled_ppi_regressor_plot_filename = plot_signal(
                downsampled_ppi_regressor_file, tr, plot_title
            )

            condition_plots.append(
                {
                    "name": cond_name,
                    "upsampled_regressor_plot": embed_image(
                        upsampled_condition_regressor_plot_filename
                    ),
                    "upsampled_ppi_plot": embed_image(upsampled_ppi_plot_filename),
                    "downsampled_ppi_plot": embed_image(
                        downsampled_ppi_regressor_plot_filename
                    ),
                }
            )

        report.add_context(
            condition_names=condition_names,
            condition_plots=condition_plots,
        )

        smoothed_nifti_file = perform_spatial_smoothing(
            subject_dir.parent,
            afni_img_path,
            percent_change_nifti_file,
            mask_file,
            fwhm,
        )

        get_task_deconvolve_cmd = {
            "kids": get_task_deconvolve_kids_cmd,
            "adults": get_task_deconvolve_adults_cmd,
        }

        deconvolve_cmd = get_task_deconvolve_cmd[cohort](
            task, timing_dir, nuisance_regressors_file, seed_timeseries_file, ppi_dir
        )

        report.add_context(
            deconvolve_cmd=f"{deconvolve_cmd['num_stimts']} {deconvolve_cmd['args']}",
        )

        design_matrix_file = create_design_matrix(
            subject_dir.parent,
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
            stats_file_relm,
            betas_dir,
            afni_img_path,
            cohort,
            task,
            analysis_type="gPPI",
        )

        report.create_report(report_path, "first_level.html")


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
