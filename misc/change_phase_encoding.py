import argparse, json

from bids import BIDSLayout


def _get_cmd_args():
    parser = argparse.ArgumentParser(description="Change phase encoding direction.")
    parser.add_argument(
        "--bids_dir",
        dest="bids_dir",
        required=True,
        help="The root of the BIDS directory.",
    )
    parser.add_argument(
        "--phase_encoding_direction",
        dest="phase_encoding_direction",
        default="j-",
        required=False,
        help="The phase encoding direction.",
    )

    return parser


def main(bids_dir, phase_encoding_direction):
    layout = BIDSLayout(bids_dir)
    nifti_files = layout.get(
        return_type="file",
        target="subject",
        suffix="bold",
        extension=".nii.gz",
        scope="raw",
    )

    for nifti_file in nifti_files:
        json_file = str(nifti_file).replace(".nii.gz", ".json")
        with open(json_file, "r") as f:
            json_data = json.load(f)

        with open(json_file, "w") as f:
            json_data["PhaseEncodingDirection"] = phase_encoding_direction
            json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
