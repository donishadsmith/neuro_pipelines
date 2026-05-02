import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from _general_utils import _convert_to_bool
from get_connors_score import run_pipeline


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extracts information from Conners 4 PDF files."
    )
    parser.add_argument(
        "--pdf_dir",
        dest="pdf_dir",
        required=True,
        help="Absolute path of the directory containing the Connors 4 PDF data.",
    )
    parser.add_argument(
        "--csv_file_path",
        dest="csv_file_path",
        default=None,
        required=False,
        help="File path for CSV file containing Conners 4 data. If CSV file exists.",
    )
    parser.add_argument(
        "--include_assessment_dates",
        dest="include_assessment_dates",
        default=True,
        required=False,
        type=_convert_to_bool,
        help="Include the date of Connors 4 was issued to participant.",
    )

    return parser


if __name__ == "__main__":
    option = _get_parser().parse_args()
    kwargs = vars(option)
    run_pipeline(**kwargs)
