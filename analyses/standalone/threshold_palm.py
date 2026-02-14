"""Standolone file for thresholding Palm outputs"""

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nifti2bids.logging import setup_logger
from _utils import get_task_contrasts, threshold_palm_output

LGR = setup_logger(__name__)


def _get_cmd_args():
    parser = argparse.ArgumentParser(description="Threshold images.")
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=True,
        help="Root path to directory containing the PALM permutation files.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="The destination directory for analysis.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        default="mph",
        required=False,
        help="Name of dataset.",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        default="kids",
        required=False,
        help="The cohort.",
    )
    parser.add_argument(
        "--cluster_correction_p",
        dest="cluster_correction_p",
        default=0.05,
        type=float,
        required=False,
        help=(
            "Significance threshold for cluster significance for nonparametric. "
            "This script uses the threshold free cluster enhancement approach which "
            "eliminates the need to select an arbritrary threshold for the voxels (cluster-forming threshold)"
        ),
    )
    parser.add_argument("--task", dest="task", required=True, help="Name of the task.")
    parser.add_argument(
        "--analysis_type",
        dest="analysis_type",
        required=True,
        help="The type of analysis performed (glm or gPPI).",
    )

    return parser


def create_glt_dict(dataset, cohort):
    # Positive codes are the anchors and the only ones that need to be created
    glt_codes_dict = {"positive": None}
    if dataset == "mph" and cohort == "kids":
        glt_codes_dict["positive"] = ["5_vs_0", "10_vs_0", "10_vs_5"]

    return glt_codes_dict


def get_output_prefixes(analysis_dir, task, contrast):
    # Positive files are the anchors and the only ones that need to be retrieved
    output_prefixes = {"positive": None}
    prefix = f"task-{task}_contrast-{contrast}_desc-nonparametric_positive"
    files = list(analysis_dir.rglob(f"*{prefix}*"))
    output_prefixes["positive"] = files[0].parent / prefix
    output_prefixes["negative"] = files[0].parent / str(prefix).replace(
        "positive", "negative"
    )

    return output_prefixes, len(
        [file for file in files if "vox_tstat_fwep" in str(file)]
    )


def main(
    analysis_dir, dst_dir, dataset, cohort, task, analysis_type, cluster_correction_p
):
    analysis_dir = Path(analysis_dir)
    dst_dir = Path(dst_dir)

    LGR.info(f"TASK: {task}")

    contrasts = get_task_contrasts(task, analysis_type, caller="threshold_palm_images")
    for contrast in contrasts:
        output_prefixes, n_contrasts = get_output_prefixes(analysis_dir, task, contrast)
        glt_codes_dict = create_glt_dict(dataset, cohort)
        glt_codes_dict["positive"] = list(glt_codes_dict["positive"])[:n_contrasts]

        threshold_palm_output(
            output_prefixes, glt_codes_dict, cluster_correction_p, dst_dir
        )


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
