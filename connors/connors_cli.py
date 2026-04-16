import argparse

from get_connors_score import run_pipeline


def _get_parser() -> argparse.ArgumentParser:
    """Returns arguments passed to Python scripts."""
    parser = argparse.ArgumentParser(
        description="Extracts information from Conners 4 PDF files."
    )

    parser.add_argument(
        "--pdf_dir",
        required=True,
        help="Absolute path of the directory containing the Connors 4 PDF data.",
    )

    parser.add_argument(
        "--csv_file_path",
        default=None,
        required=False,
        help="File path for CSV file containing Conners 4 data. If CSV file exists.",
    )

    parser.add_argument(
        "--subjects",
        default=None,
        required=False,
        help="Restrict extraction to specific subject IDs.",
    )

    return parser


if __name__ == "__main__":
    option = _get_parser().parse_args()
    kwargs = vars(option)
    run_pipeline(**kwargs)
