import argparse
from pathlib import Path

from nifti2bids.audit import BIDSAuditor


def _get_cmd_args():
    parser = argparse.ArgumentParser(description="Audit files.")
    parser.add_argument(
        "--bids_dir",
        dest="bids_dir",
        required=True,
        help="The root of the BIDS directory.",
    )
    parser.add_argument(
        "--deriv_dir",
        dest="deriv_dir",
        required=False,
        default=None,
        help="Root of derivatives directory.",
    )
    parser.add_argument(
        "--analysis_dir",
        dest="analysis_dir",
        required=False,
        default=None,
        help="Root of analysis directory.",
    )
    parser.add_argument(
        "--out_dir",
        dest="out_dir",
        help="Output directory for CSV files.",
    )

    return parser


def main(bids_dir, deriv_dir, analysis_dir, out_dir):
    auditor = BIDSAuditor(bids_dir, deriv_dir)

    df_dict = {
        "raw_niftis": auditor.check_raw_nifti_availability(),
        "raw_sidecars": auditor.check_raw_sidecar_availability(),
        "event_timing": auditor.check_events_availability(),
    }

    if deriv_dir:
        df_dict.update(
            {"preprocessed_niftis": auditor.check_preprocessed_nifti_availability()}
        )

    if analysis_dir:
        df_dict.update(
            {"first_level": auditor.check_first_level_availability(analysis_dir)}
        )

    for key in df_dict:
        df_dict[key].to_csv(Path(out_dir) / f"{key}.csv", sep=",", index=False)


if __name__ == "__main__":
    _get_cmd_args = _get_cmd_args()
    args = _get_cmd_args.parse_args()
    main(**vars(args))
