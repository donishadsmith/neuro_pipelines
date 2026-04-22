"Shared utilities"

from pathlib import Path
import shutil, subprocess

import nibabel as nib, numpy as np
from nilearn.image import new_img_like, resample_to_img

from bidsaid.files import get_entity_value
from bidsaid.logging import setup_logger
from bidsaid.metadata import needs_resampling

LGR = setup_logger(__name__)

VALID_TASK_NAMES = {
    "kids": ["nback", "mtlr", "mtle", "flanker", "princess"],
    "adults": ["nback", "mtlr", "mtle", "flanker", "simplegng", "complexgng"],
}

TASK_CONTRASTS = {
    "kids": {
        "nback": (
            "1-back_vs_center",
            "2-back_vs_center",
            "2-back_vs_1-back",
        ),
        "mtle": ("neutral_encoding",),
        "mtlr": ("neutral_retrieval",),
        "princess": ("switch_vs_nonswitch",),
        "flanker": (
            "incongruent_vs_congruent",
            "nogo_vs_neutral",
            "nogo",
            "incongruent",
            "congruent",
        ),
    },
    "adults": {
        "nback": ("2-back_vs_0-back",),
        "mtle": (
            "aversive_encoding_vs_neutral_encoding",
            "neutral_encoding",
        ),
        "mtlr": (
            "aversive_retrieval_vs_neutral_retrieval",
            "neutral_retrieval",
        ),
        "flanker": (
            "incongruent_vs_congruent",
            "nogo_vs_neutral",
            "nogo",
            "incongruent",
            "congruent",
        ),
        "simplegng": ("nogo_vs_go",),
        "complexgng": ("nogo_vs_go",),
    },
}

CONTRAST_CODES = {
    "kids": ("0", "5", "10", "5_vs_0", "10_vs_0", "10_vs_5", "mean"),
    "adults": ("mph", "placebo", "mph_vs_placebo", "15_vs_10", "mean"),
}

BETWEEN_GROUP_DOSE_CODES = {
    "adults": {"15_vs_10": "dose_mg"},
}


def get_between_group_code(cohort):
    return list(BETWEEN_GROUP_DOSE_CODES.get(cohort))[0]


def is_between_group_dose_code(second_level_glt_code, cohort):
    """
    The between contrast is only for the adult cohort. For the kids cohort,
    the design is purely within, every subject receives the 0, 5, and 10 mg,
    randomized for each visit. The adult cohort came for two visits, all adults
    received placebo; however, half the adults received 10 mg mph and the other
    half 15 mg mph. To assess dose dependent differences, each adult's mph
    data is subtracted from their placebo, to identify the BOLD differences
    that can be reasonably attributed to mph, then the difference maps
    are subjected to a between group analysis. This is not done for the kids
    because the placebo (0 mg), cancels in the contrast (e.g. for each kid
    [10 - placebo] - [5 - placebo])
    """
    if cohort != "adults":
        return False

    return second_level_glt_code in BETWEEN_GROUP_DOSE_CODES.get(cohort, {})


def in_between_group_code(second_level_glt_code, cohort):
    if cohort != "adults":
        return False

    return second_level_glt_code in list(BETWEEN_GROUP_DOSE_CODES.get(cohort, ""))[0]


def get_between_group_column(second_level_glt_code, cohort):
    return BETWEEN_GROUP_DOSE_CODES.get(cohort, {}).get(second_level_glt_code)


def get_first_level_gltsym_codes(cohort, task, analysis_type, caller):
    contrasts = TASK_CONTRASTS[cohort][task]
    if analysis_type == "gPPI":
        contrasts = modify_contrast_names(contrasts)

    return (
        (f"{contrast}#0_Coef" for contrast in contrasts)
        if caller == "extract_betas"
        else contrasts
    )


def get_second_level_glt_codes(cohort, add_dose_mg_groups=False):
    if cohort == "adults" and add_dose_mg_groups:
        codes = list(CONTRAST_CODES[cohort]) + ["10", "15"]
        return codes

    return CONTRAST_CODES[cohort]


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


def get_beta_names(gltsyms, add_coef_str=False, create_sub_conditions=True):
    if isinstance(gltsyms, str):
        gltsyms = [gltsyms]

    if not create_sub_conditions:
        return gltsyms

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


def get_contrast_name_from_file(filename):
    filename = Path(filename).name

    return filename.split("desc-")[-1].split("_betas")[0]


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
        suffix = "".join(seed_mask_path.suffixes[3:])
        possible_coordinate = possible_coordinate.removesuffix(suffix)
        if replace_underscore:
            possible_coordinate = possible_coordinate.replace("_", ",")

    return possible_coordinate


def create_beta_files(
    stats_file,
    beta_dir,
    afni_img_path,
    cohort,
    task,
    analysis_type,
    out_dir=None,
    overwrite=True,
):
    first_level_gltsyms = get_first_level_gltsym_codes(
        cohort, task, analysis_type, caller="extract_betas"
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

    curr_dir = Path.cwd()
    for filename in ["3dFWHMx.1D", "3dFWHMx.1D.png"]:
        curr_filename = curr_dir / filename
        if curr_filename.exists():
            curr_filename.unlink()

    # Use -acf for more accurate false positive rate center for fMRI data
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
    try:
        positive_tstat_file = Path(f"{output_prefix}_vox_tstat_c1.nii.gz")
        positive_pval_file = Path(f"{output_prefix}_tfce_tstat_cfwep_c1.nii.gz")
        positive_tstat_img = nib.load(positive_tstat_file)
        positive_sig_mask = (
            nib.load(positive_pval_file).get_fdata() > logp_threshold
        ).astype(float)
        positive_masked_tstat = positive_tstat_img.get_fdata() * positive_sig_mask

        # Reverse direction (e.g., 0_vs_5)
        negative_tstat_file = Path(f"{output_prefix}_vox_tstat_c2.nii.gz")
        negative_pval_file = Path(f"{output_prefix}_tfce_tstat_cfwep_c2.nii.gz")
        negative_tstat_img = nib.load(negative_tstat_file)
        negative_sig_mask = (
            nib.load(negative_pval_file).get_fdata() > logp_threshold
        ).astype(float)
        negative_masked_tstat = (
            negative_tstat_img.get_fdata() * negative_sig_mask
        ) * -1

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
    except FileNotFoundError:
        LGR.critical(
            f"For the {second_level_glt_code} code, a file was not found for the following prefix: {output_prefix}",
            exc_info=True,
        )


def get_nontarget_dose(second_level_glt_code, cohort):
    if second_level_glt_code == "mean":
        return None

    if is_between_group_dose_code(second_level_glt_code, cohort):
        return ["placebo"]

    doses = {"kids": {"0", "5", "10"}, "adults": {"mph", "placebo"}}

    return list(doses[cohort].difference(second_level_glt_code.split("_vs_")))


def drop_dose_rows(data_table, dose_list, only_paired_data=False):
    if not dose_list:
        return data_table

    data_table = data_table[~data_table["dose"].astype(str).isin(dose_list)]
    if only_paired_data:
        # Keep only subjects who have both remaining doses (i.e., appear more than once)
        duplicated_mask = data_table["Subj"].duplicated(keep=False)
        if not duplicated_mask.to_numpy().all():
            contrast_name = get_contrast_name_from_file(
                data_table["InputFile"].tolist()[0]
            )
            removed_subjects = data_table.loc[~duplicated_mask, "Subj"].tolist()
            total_subjects = len(
                set(data_table["Subj"].tolist()).difference(removed_subjects)
            )
            LGR.warning(
                f"For contrast ({contrast_name}), the following subjects have been removed: {removed_subjects}. "
                f"A total of {total_subjects} unique subjects with two timepoints remain."
            )

        return data_table[duplicated_mask]

    return data_table


def get_group_labels(second_level_glt_code):
    return second_level_glt_code.split("_vs_")


def save_binary_mask(mask_img_fdata, affine, hdr, mask_filename):
    """To save as a True binary mask and prevent equality index due to floating point issues"""
    mask_img_fdata = mask_img_fdata.astype(np.int8)
    hdr.set_data_dtype(np.int8)

    mask_img = nib.nifti1.Nifti1Image(mask_img_fdata, affine, hdr)

    nib.save(mask_img, mask_filename)
