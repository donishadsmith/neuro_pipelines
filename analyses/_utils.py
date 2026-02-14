"Shared utilities"

from pathlib import Path
import shutil, subprocess

import nibabel as nib, numpy as np
from nilearn.image import new_img_like

from nifti2bids.bids import get_entity_value
from nifti2bids.logging import setup_logger

LGR = setup_logger(__name__)
LGR.setLevel("INFO")


def get_task_contrasts(task, caller):
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
            "congruent_vs_incongruent",
            "congruent_vs_nogo",
            "incongruent_vs_nogo",
        )

    return (
        (f"{contrast}#0_Coef" for contrast in contrasts)
        if caller == "extract_betas"
        else contrasts
    )


def create_contrast_files(
    stats_file, contrast_dir, afni_img_path, task, out_dir=None, overwrite=True
):
    contrasts = get_task_contrasts(task, caller="extract_betas")

    for contrast in contrasts:
        contrast_file = contrast_dir / stats_file.name.replace(
            "stats", contrast.replace("#0_Coef", "_betas")
        )
        if contrast_file.exists() and overwrite:
            contrast_file.unlink()

        cmd = (
            f"apptainer exec -B /projects:/projects {afni_img_path} 3dbucket "
            f"{stats_file}'[{contrast}]' "
            f"-prefix {contrast_file} "
            "-overwrite"
        )
        LGR.info(f"Extracting {contrast} contrast: {cmd}")

        try:
            subprocess.run(cmd, shell=True, check=True)
        except Exception:
            LGR.critical(f"The following command failed: {cmd}", exc_info=True)

        if out_dir and contrast_file.exists():
            path = Path(out_dir) / contrast_file.name
            if path.exists():
                LGR.info("Replacing old file with new file.")
                path.unlink()

            shutil.move(contrast_file, out_dir)


def get_number_of_censored_volumes(censored_filename):
    arr = np.loadtxt(censored_filename)
    arr = arr.astype(int)

    return arr[arr == 0].size


def estimate_noise_smoothness(
    analysis_dir, afni_img_path, group_mask_filename, residual_filename, contrast
):
    task = get_entity_value(group_mask_filename.name, "task")
    acf_parameters_filename = (
        analysis_dir
        / "acf_parameters"
        / f"task-{task}_contrast-{contrast}_desc-acf_parameters.txt"
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
    analysis_dir, afni_img_path, group_mask_filename, acf_parameters_filename, contrast
):
    task = get_entity_value(group_mask_filename.name, "task")
    # Partial filename
    output_filename_prefix = (
        analysis_dir
        / "cluster_correction"
        / f"task-{task}_contrast-{contrast}_desc-cluster_correction"
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


def replace_extension(filename, new_ext):
    filename = Path(filename)
    old_ext = "".join(filename.suffixes)

    return Path(str(filename).replace(old_ext, new_ext))


def threshold_palm_output(
    output_prefixes, glt_codes_dict, cluster_correction_p, dst_dir
):
    logp_threshold = -np.log10(cluster_correction_p)
    LGR.info(
        f"Using -log10(p) threshold: {logp_threshold:.4f} "
        f"(cluster_significance={cluster_correction_p})"
    )

    output_dir = output_prefixes["positive"].parent
    prefix_positive = output_prefixes["positive"].name
    prefix_negative = output_prefixes["negative"].name

    # If only one contrast, palm excludes c{index}; however
    # a minimum of two contrasts are needed since only one tail will
    # be used
    for index, glt_code in enumerate(glt_codes_dict["positive"], 1):
        LGR.info(f"Processing contrasts for {index}: {glt_code}")

        # Forward direction (e.g., 5_vs_0)
        positive_tstat_file = (
            output_dir / f"{prefix_positive}_tfce_tstat_c{index}.nii.gz"
        )
        if not positive_tstat_file.exists():
            positive_tstat_file = replace_extension(positive_tstat_file, ".nii")

        positive_pval_file = (
            output_dir / f"{prefix_positive}_tfce_tstat_fwep_c{index}.nii.gz"
        )
        if not positive_pval_file.exists():
            positive_pval_file = replace_extension(positive_pval_file, ".nii")

        positive_tstat_img = nib.load(positive_tstat_file)
        positive_sig_mask = (
            nib.load(positive_pval_file).get_fdata() > logp_threshold
        ).astype(float)
        positive_masked_tstat = positive_tstat_img.get_fdata() * positive_sig_mask

        # Reverse direction (e.g., 0_vs_5)
        negative_tstat_file = (
            output_dir / f"{prefix_negative}_tfce_tstat_c{index}.nii.gz"
        )
        if not negative_tstat_file.exists():
            negative_tstat_file = replace_extension(negative_tstat_file, ".nii")

        negative_pval_file = (
            output_dir / f"{prefix_negative}_tfce_tstat_fwep_c{index}.nii.gz"
        )
        if not negative_pval_file.exists():
            negative_pval_file = replace_extension(negative_pval_file, ".nii")

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
        combined_thresholded_file = (
            dst_dir / f"task-{prefix_positive.split('task-')[1].split('_contrast')[0]}_"
            f"contrast-{prefix_positive.split('contrast-')[1].split('_desc')[0]}_"
            f"gltcode-{glt_code}_desc-nonparametric_thresholded_bisided.nii.gz"
        )
        nib.save(combined_thresholded_img, combined_thresholded_file)

        LGR.info(f"Saved thresholded t-map: {combined_thresholded_file}")
