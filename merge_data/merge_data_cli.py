import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from merge_data import run_pipeline


def _get_cmd_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge columns from a secondary file into a primary file. "
            "Rows are matched on subject ID, and optionally on date and dose."
        )
    )
    parser.add_argument(
        "--primary_file",
        dest="primary_file",
        required=True,
        help="Path to file to use as the base file to receive new columns.",
    )
    parser.add_argument(
        "--primary_file_subject_column",
        dest="primary_file_subject_column",
        required=True,
        help="Name of the column containing the subject IDs in the primary file.",
    )
    parser.add_argument(
        "--primary_file_date_column",
        dest="primary_file_date_column",
        required=False,
        default=None,
        help=(
            "Name of the column containing the dates in the primary file. "
            "Omit for data that does not vary by visit (e.g., demographics)."
        ),
    )
    parser.add_argument(
        "--primary_file_dose_column",
        dest="primary_file_dose_column",
        required=False,
        default=None,
        help="Name of the column containing the doses in the primary file.",
    )
    parser.add_argument(
        "--secondary_file",
        dest="secondary_file",
        required=True,
        help="Path to the file the new columns will be received from.",
    )
    parser.add_argument(
        "--secondary_file_subject_column",
        dest="secondary_file_subject_column",
        required=True,
        help="Name of the column containing the subject IDs in the secondary file.",
    )
    parser.add_argument(
        "--secondary_file_date_column",
        dest="secondary_file_date_column",
        required=False,
        default=None,
        help=(
            "Name of the column containing the dates in the secondary file. "
            "Must be given together with --primary_file_date_column."
        ),
    )
    parser.add_argument(
        "--secondary_file_dose_column",
        dest="secondary_file_dose_column",
        required=False,
        default=None,
        help="Name of the column containing the doses in the secondary file.",
    )
    parser.add_argument(
        "--columns_to_add",
        dest="columns_to_add",
        required=True,
        nargs="+",
        help="Column names from the secondary file to add to the primary file.",
    )
    parser.add_argument(
        "--column_suffix",
        dest="column_suffix",
        required=False,
        default=None,
        help=(
            "Suffix appended to every added column (e.g., '_5_vs_0_superior_parietal'). "
            "Use this when the secondary file shares column names with the primary file."
        ),
    )

    return parser


if __name__ == "__main__":
    args = _get_cmd_args().parse_args()

    if (args.primary_file_date_column is None) != (
        args.secondary_file_date_column is None
    ):
        raise SystemExit(
            "Both --primary_file_date_column and --secondary_file_date_column "
            "must be given to merge on date."
        )

    if (args.primary_file_dose_column is None) != (
        args.secondary_file_dose_column is None
    ):
        raise SystemExit(
            "Both --primary_file_dose_column and --secondary_file_dose_column "
            "must be given to merge on dose."
        )

    run_pipeline(**vars(args))
