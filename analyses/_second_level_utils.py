import subprocess
from pathlib import Path

import nibabel as nib, numpy as np
from bidsaid.files import get_entity_value
from bidsaid.logging import setup_logger
from nilearn.image import new_img_like

from _utils import get_contrast_entity_key

LGR = setup_logger(__name__)


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
        / "second_level"
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

    # Use -acf for more accurate false positive rate
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

    return acf_parameters_filename


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
