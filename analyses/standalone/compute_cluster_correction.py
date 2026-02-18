import argparse, sys
from pathlib import Path

from nifti2bids.logging import setup_logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _utils import (
    get_contrast_entity_key,
    get_first_level_gltsym_codes,
    estimate_noise_smoothness,
    perform_cluster_simulation,
)

LGR = setup_logger(__name__)


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
    parser.add_argument(
        "--analysis_type",
        dest="analysis_type",
        required=True,
        choices=["glm", "gPPI"],
        help="The type of analysis performed (glm or gPPI).",
    )

    return parser


def main(analysis_dir, afni_img_path, task, analysis_type):
    analysis_dir = Path(analysis_dir)

    LGR.info(f"TASK: {task}")

    first_level_gltlabels = get_first_level_gltsym_codes(
        task, analysis_type, caller="compute_cluster_correction"
    )
    for first_level_gltlabel in first_level_gltlabels:
        entity_key = get_contrast_entity_key(first_level_gltlabel)
        LGR.info(f"FIRST LEVEL GLTLABEL: {first_level_gltlabel}")

        group_mask_filename = next(
            analysis_dir.rglob(
                f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-group_mask.nii.gz"
            )
        )
        residual_filename = next(
            analysis_dir.rglob(
                f"task-{task}_{entity_key}-{first_level_gltlabel}_desc-parametric_residuals.nii.gz"
            )
        )

        acf_parameters_filename = estimate_noise_smoothness(
            analysis_dir,
            afni_img_path,
            group_mask_filename,
            residual_filename,
            first_level_gltlabel,
        )

        perform_cluster_simulation(
            analysis_dir,
            afni_img_path,
            group_mask_filename,
            acf_parameters_filename,
            first_level_gltlabel,
        )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
