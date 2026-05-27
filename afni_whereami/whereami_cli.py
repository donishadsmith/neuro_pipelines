import argparse

from whereami import run_pipeline


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Pipeline for looking up MNI coordinates with AFNI's whereami."
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

    return parser


if __name__ == "__main__":
    _get_cmd_args = _get_cmd_args()
    args = _get_cmd_args.parse_args()
    output_text, _, _ = run_pipeline(**vars(args))
    print(output_text)
