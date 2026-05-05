import argparse

from sphere_mask import run_pipeline


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Pipeline for generating a-priori sphere masks for seed-based connectivity analyses."
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=True,
        choices=["kids", "adults"],
        help="The name of the cohort.",
    )
    parser.add_argument(
        "--mni_coordinate",
        dest="mni_coordinate",
        required=True,
        nargs=3,
        help="The MNI coordinate.",
    )
    parser.add_argument(
        "--sphere_radius",
        dest="sphere_radius",
        required=True,
        type=float,
        help="The radius of the sphere mask in mm.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=False,
        default=None,
        help="The root of the output directory for the sphere mask and plot.",
    )

    return parser


if __name__ == "__main__":
    _get_cmd_args = _get_cmd_args()
    args = _get_cmd_args.parse_args()
    run_pipeline(**vars(args))
