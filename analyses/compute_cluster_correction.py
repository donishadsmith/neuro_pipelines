import argparse, subprocess
from pathlib import Path

from nifti2bids.bids import get_entity_value
from nifti2bids.logging import setup_logger

from _utils import get_task_contrasts

LGR = setup_logger(__name__)
LGR.setLevel("INFO")


def _get_cmd_args():
    parser = argparse.ArgumentParser(description="Compute cluster threshold stats.")
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=True,
        help="Path to directory containing second level results.",
    )
    parser.add_argument(
        "--afni_img_path",
        dest="afni_img_path",
        required=True,
        help="Path to Singularity image of Afni with R.",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")

    return parser


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
        f"singularity exec --no-home -B /projects:/projects {afni_img_path} 3dFWHMx "
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
        f"singularity exec --no-home -B /projects:/projects {afni_img_path} 3dClustSim "
        f"-mask {group_mask_filename} "
        f"-prefix {output_filename_prefix} "
        f"-acf $(awk 'NR == 2 {{print $1, $2, $3}}' {acf_parameters_filename})"
    )

    LGR.info(f"Running 3dClustSim: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main(analysis_dir, afni_img_path, task):
    analysis_dir = Path(analysis_dir)

    LGR.info(f"TASK: {task}")

    contrasts = get_task_contrasts(task, caller="compute_cluster_correction")
    for contrast in contrasts:
        LGR.info(f"CONTRAST: {contrast}")

        group_mask_filename = next(
            analysis_dir.rglob(
                f"task-{task}_contrast-{contrast}_desc-group_mask.nii.gz"
            )
        )
        residual_filename = next(
            analysis_dir.rglob(f"task-{task}_contrast-{contrast}_desc-residuals.nii.gz")
        )

        acf_parameters_filename = estimate_noise_smoothness(
            analysis_dir,
            afni_img_path,
            group_mask_filename,
            residual_filename,
            contrast,
        )

        perform_cluster_simulation(
            analysis_dir,
            afni_img_path,
            group_mask_filename,
            acf_parameters_filename,
            contrast,
        )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
