import argparse, sys
from pathlib import Path

from bidsaid.logging import setup_logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _utils import (
    get_contrast_entity_key,
    get_first_level_gltsym_codes,
    get_between_group_code,
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
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination directory for analysis.",
    )
    parser.add_argument(
        "--afni_img_path",
        dest="afni_img_path",
        required=True,
        help="Path to Apptainer image of Afni with R.",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=True,
        choices=["adults", "kids"],
        help="The cohort to analyze.",
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


def main(analysis_dir, dst_dir, afni_img_path, cohort, task, analysis_type):
    analysis_dir = Path(analysis_dir)
    dst_dir = Path(dst_dir)

    LGR.info(f"TASK: {task}")

    first_level_glt_labels = get_first_level_gltsym_codes(
        cohort, task, analysis_type, caller="compute_cluster_correction"
    )
    for first_level_glt_label in first_level_glt_labels:
        entity_key = get_contrast_entity_key(first_level_glt_label)
        LGR.info(f"FIRST LEVEL GLTLABEL: {first_level_glt_label}")

        group_mask_filename = next(
            analysis_dir.rglob(
                f"task-{task}_{entity_key}-{first_level_glt_label}_desc-parametric_group_mask.nii.gz"
            )
        )
        residual_filename = next(
            analysis_dir.rglob(
                f"task-{task}_{entity_key}-{first_level_glt_label}_desc-parametric_residuals.nii.gz"
            )
        )

        acf_parameters_filename = estimate_noise_smoothness(
            dst_dir,
            afni_img_path,
            group_mask_filename,
            residual_filename,
            first_level_glt_label,
        )

        perform_cluster_simulation(
            afni_img_path,
            group_mask_filename,
            acf_parameters_filename,
            first_level_glt_label,
        )

        if cohort == "adults":
            between_group_code = get_between_group_code(cohort)
            group_mask_filename = next(
                analysis_dir.rglob(
                    f"task-{task}_{entity_key}-{first_level_glt_label}_gltcode-{between_group_code}_desc-parametric_group_mask.nii.gz"
                )
            )
            residual_filename = next(
                analysis_dir.rglob(
                    f"task-{task}_{entity_key}-{first_level_glt_label}__gltcode-{between_group_code}_desc-parametric_residuals.nii.gz"
                )
            )

            acf_parameters_filename = estimate_noise_smoothness(
                dst_dir,
                afni_img_path,
                group_mask_filename,
                residual_filename,
                first_level_glt_label,
            )

            perform_cluster_simulation(
                afni_img_path,
                group_mask_filename,
                acf_parameters_filename,
                first_level_glt_label,
            )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
