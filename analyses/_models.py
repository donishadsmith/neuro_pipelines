import subprocess
from pathlib import Path

from nifti2bids.logging import setup_logger

LGR = setup_logger(__name__)


def create_design_matrix(
    subject_dir,
    afni_img_path,
    smoothed_nifti_file,
    mask_file,
    censor_file,
    contrast_cmd,
    cosine_regressor_names,
):
    design_matrix_file = subject_dir / str(smoothed_nifti_file).replace(
        "smoothed.nii.gz", "design_matrix.1D"
    )

    polort = 0 if cosine_regressor_names else 4
    LGR.info(f"Using polort {polort} for 3dDeconvolve.")

    cmd = (
        f"apptainer exec -B /projects:/projects {afni_img_path} 3dDeconvolve "
        f"-input {smoothed_nifti_file} "
        f"-mask {mask_file} "
        f"-censor {censor_file} "
        f"-polort {polort} "
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
