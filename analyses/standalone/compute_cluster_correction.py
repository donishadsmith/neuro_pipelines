import argparse
from pathlib import Path

from nifti2bids.logging import setup_logger

from _utils import (
    get_task_contrasts,
    estimate_noise_smoothness,
    perform_cluster_simulation,
)

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
        help="Path to Apptainer image of Afni with R.",
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")

    return parser


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
