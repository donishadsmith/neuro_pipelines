"Shared utilities"

from pathlib import Path
import shutil, subprocess

import nibabel as nib, numpy as np
from nilearn.image import new_img_like, resample_to_img

from nifti2bids.bids import get_entity_value
from nifti2bids.logging import setup_logger
from nifti2bids.metadata import needs_resampling
from nifti2bids.io import replace_ext

LGR = setup_logger(__name__)


def get_first_level_gltsym_codes(task, analysis_type, caller):
    if task == "nback":
        contrasts = (
            "1-back_vs_0-back",
            "2-back_vs_0-back",
            "2-back_vs_1-back",
        )
    elif task == "mtle":
        contrasts = ("indoor",)
    elif task == "mtlr":
        contrasts = ("seen",)
    elif task == "princess":
        contrasts = ("switch_vs_nonswitch",)
    else:
        contrasts = (
            "congruent_vs_neutral",
            "incongruent_vs_neutral",
            "nogo_vs_neutral",
            "incongruent_vs_congruent",
            "congruent_vs_nogo",
            "incongruent_vs_nogo",
        )

    if analysis_type == "gPPI":
        contrasts = modify_contrast_names(contrasts)

    return (
        (f"{contrast}#0_Coef" for contrast in contrasts)
        if caller == "extract_betas"
        else contrasts
    )


def get_second_level_glt_codes():
    return ["0", "5", "10", "5_vs_0", "10_vs_0", "10_vs_5", "mean"]


def modify_contrast_names(contrasts):
    modified_contrasts = []
    for contrast in contrasts:
        if "_vs_" in contrast:
            modified_contrasts.append(
                f"PPI_{contrast.split('_vs_')[0]}_vs_PPI_{contrast.split('_vs_')[1]}"
            )

        else:
            modified_contrasts.append(f"PPI_{contrast}")

    return modified_contrasts


def get_beta_names(gltsyms, add_coef_str=False):
    if isinstance(gltsyms, str):
        gltsyms = [gltsyms]

    beta_names = []
    for gltsym in gltsyms:
        names = [] if "_vs_" not in gltsym else gltsym.split("_vs_")
        names += [gltsym]

        beta_names.extend(names)

    if add_coef_str:
        beta_names = [
            f"{name}#0_Coef" if not name.endswith("#0_Coef") else name
            for name in beta_names
        ]

    return list(set(beta_names))


def get_contrast_entity_key(input_str):
    input_str = Path(input_str).name

    return "contrast" if "_vs_" in input_str else "condition"


def resample_seed_img(seed_img, subject_nifti_img):
    if needs_resampling(seed_img, subject_nifti_img):
        seed_img = resample_to_img(
            seed_img, subject_nifti_img, interpolation="nearest", copy_header=True
        )

    return seed_img


def get_coordinate_from_filename(seed_mask_path, replace_underscore=True):
    seed_mask_path = Path(seed_mask_path)
    possible_coordinate = ""
    if "_sphere_mask_" in seed_mask_path.name:
        possible_coordinate = seed_mask_path.name.split("_sphere_mask_")[1]
        suffix = "".join(seed_mask_path.suffixes)
        possible_coordinate = possible_coordinate.removesuffix(suffix)
        if replace_underscore:
            possible_coordinate = possible_coordinate.replace("_", ",")

    return possible_coordinate


def create_beta_files(
    stats_file,
    beta_dir,
    afni_img_path,
    task,
    analysis_type,
    out_dir=None,
    overwrite=True,
):
    first_level_gltsyms = get_first_level_gltsym_codes(
        task, analysis_type, caller="extract_betas"
    )
    beta_names = get_beta_names(first_level_gltsyms, add_coef_str=True)

    for beta_name in beta_names:
        beta_file = beta_dir / stats_file.name.replace(
            "stats", beta_name.replace("#0_Coef", "_betas")
        )
        if beta_file.exists() and overwrite:
            beta_file.unlink()
        cmd = (
            f"apptainer exec -B /projects:/projects {afni_img_path} 3dbucket "
            f"{stats_file}'[{beta_name}]' "
            f"-prefix {beta_file} "
            "-overwrite"
        )
        LGR.info(f"Extracting {beta_name} betas: {cmd}")

        try:
            subprocess.run(cmd, shell=True, check=True)
        except Exception:
            LGR.critical(f"The following command failed: {cmd}", exc_info=True)

        if out_dir and beta_file.exists():
            path = Path(out_dir) / beta_file.name
            if path.exists():
                LGR.info("Replacing old file with new file.")
                path.unlink()

            shutil.move(beta_file, out_dir)


def estimate_noise_smoothness(
    dst_dir,
    afni_img_path,
    group_mask_filename,
    residual_filename,
    first_level_gltlabel,
):
    task = get_entity_value(group_mask_filename.name, "task")
    entity_key = get_contrast_entity_key(group_mask_filename)
    acf_parameters_filename = (
        dst_dir
        / "second_level_outputs"
        / "parametric"
        / f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-acf_parameters.txt"
    )
    acf_parameters_filename.parent.mkdir(parents=True, exist_ok=True)
    if acf_parameters_filename.exists():
        acf_parameters_filename.unlink()

    # Use -acf for more accurate false positive rate control for fMRI data
    cmd = (
        f"apptainer exec --no-home -B /projects:/projects {afni_img_path} 3dFWHMx "
        f"-mask {group_mask_filename} "
        f"-input {residual_filename} "
        f"-acf > {acf_parameters_filename}"
    )

    LGR.info(f"Running 3dFWHMx: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    return acf_parameters_filename


def perform_cluster_simulation(
    afni_img_path,
    group_mask_filename,
    acf_parameters_filename,
    first_level_glt,
):
    task = get_entity_value(group_mask_filename.name, "task")
    entity_key = get_contrast_entity_key(group_mask_filename)
    # Partial filename
    output_filename_prefix = (
        acf_parameters_filename.parent
        / f"task-{task}_{entity_key}-{first_level_glt}_desc-cluster_correction"
    )
    output_filename_prefix.parent.mkdir(parents=True, exist_ok=True)

    curr_dir = Path.cwd()
    for filename in ("3dFWHMx.1D", "3dFWHMx.1D.png"):
        curr_filename = curr_dir / filename
        if curr_filename.exists():
            curr_filename.unlink()

    cmd = (
        f"apptainer exec --no-home -B /projects:/projects {afni_img_path} 3dClustSim "
        f"-mask {group_mask_filename} "
        f"-prefix {output_filename_prefix} "
        f"-acf $(awk 'NR == 2 {{print $1, $2, $3}}' {acf_parameters_filename})"
    )

    LGR.info(f"Running 3dClustSim: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def threshold_palm_output(output_prefix, second_level_glt_code, cluster_correction_p):
    logp_threshold = -np.log10(cluster_correction_p)
    LGR.info(
        f"Using -log10(p) threshold: {logp_threshold:.4f} "
        f"(cluster_significance={cluster_correction_p})"
    )

    # If only one contrast, palm excludes c{index}; however
    # a minimum of two images are needed since only one tail will
    # be used
    LGR.info(f"Thresholding images for the following glt code: {second_level_glt_code}")

    output_prefix = str(output_prefix).removesuffix("_")
    # Forward direction (e.g., 5_vs_0)
    positive_tstat_file = Path(f"{output_prefix}_vox_tstat_c1.nii.gz")
    if not positive_tstat_file.exists():
        positive_tstat_file = replace_ext(positive_tstat_file, ".nii")

    positive_pval_file = Path(f"{output_prefix}_tfce_tstat_fwep_c1.nii.gz")
    if not positive_pval_file.exists():
        positive_pval_file = replace_ext(positive_pval_file, ".nii")

    positive_tstat_img = nib.load(positive_tstat_file)
    positive_sig_mask = (
        nib.load(positive_pval_file).get_fdata() > logp_threshold
    ).astype(float)
    positive_masked_tstat = positive_tstat_img.get_fdata() * positive_sig_mask

    # Reverse direction (e.g., 0_vs_5)
    negative_tstat_file = Path(f"{output_prefix}_vox_tstat_c2.nii.gz")
    if not negative_tstat_file.exists():
        negative_tstat_file = replace_ext(negative_tstat_file, ".nii")

    negative_pval_file = Path(f"{output_prefix}_tfce_tstat_fwep_c2.nii.gz")
    if not negative_pval_file.exists():
        negative_pval_file = replace_ext(negative_pval_file, ".nii")

    negative_tstat_img = nib.load(negative_tstat_file)
    negative_sig_mask = (
        nib.load(negative_pval_file).get_fdata() > logp_threshold
    ).astype(float)
    negative_masked_tstat = (negative_tstat_img.get_fdata() * negative_sig_mask) * -1

    # Combine, significant clusters should not overlap/ mutually exclusive
    combined_masked_tstat = positive_masked_tstat + negative_masked_tstat
    combined_thresholded_img = new_img_like(
        positive_tstat_img,
        combined_masked_tstat,
        affine=positive_tstat_img.affine,
        copy_header=True,
    )

    # Use glt_code in filename (e.g., 5_vs_0)
    combined_thresholded_file = f"{output_prefix}_thresholded_bisided.nii.gz"
    nib.save(combined_thresholded_img, combined_thresholded_file)
    LGR.info(f"Saved thresholded t-map: {combined_thresholded_file}")


def get_nontarget_dose(second_level_glt_code):
    if second_level_glt_code == "mean":
        return None

    return list(
        {"0", "5", "10"}.difference(
            second_level_glt_code.replace("PPI_", "").split("_vs_")
        )
    )


def drop_dose_rows(data_table, dose_list, only_paired_data=False):
    if not dose_list:
        return data_table

    data_table = data_table[~data_table["dose"].astype(str).isin(dose_list)]
    if only_paired_data:
        # Keep only subjects who have both remaining doses (i.e., appear more than once)
        return data_table[data_table["participant_id"].duplicated(keep=False)]

    return data_table


def get_interpretation_labels(second_level_glt_code):
    return second_level_glt_code.split("_vs_")
