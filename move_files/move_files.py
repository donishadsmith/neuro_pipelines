import argparse, shutil
from pathlib import Path

from nifti2bids.bids import get_entity_value


def _get_cmd_args():
    parser = argparse.ArgumentParser(
        description="Move events or sessions files to BIDS directory."
    )
    parser.add_argument(
        "--src_dir",
        dest="src_dir",
        required=True,
        help=("Path containing the files. File naming should be BIDS compliant."),
    )
    parser.add_argument(
        "--bids_dir",
        dest="bids_dir",
        required=True,
        help="The root of the BIDS directory.",
    )

    return parser


def main(src_dir, bids_dir):
    src_dir = Path(src_dir)
    bids_dir = Path(bids_dir)

    for src_file in src_dir.rglob("*"):
        if not src_file.name.startswith("sub-"):
            continue

        subject = get_entity_value(src_file, "sub", return_entity_prefix=True)
        session = get_entity_value(src_file, "ses", return_entity_prefix=True)

        dst_dir = (
            bids_dir
            / subject
            / (session or "")
            / ("" if src_file.name.endswith("sessions.tsv") else "func")
        )

        try:
            dst_file = dst_dir / src_file.name
            if dst_file.exists():
                dst_file.unlink()

            shutil.move(str(src_file), dst_dir)
        except:
            pass


if __name__ == "__main__":
    _get_cmd_args = _get_cmd_args()
    args = _get_cmd_args.parse_args()
    main(**vars(args))
