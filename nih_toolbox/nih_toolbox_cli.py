import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "nih_toolbox"))

from organize_toolbox_data import run_pipeline


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Organize the NIH toolbox data (pivot from long to wide form). "
            "Use the CSV file containing the following columns: 'RawScore', "
            "'Theta', 'SE', 'TScore', 'Computed Score', 'Uncorrected Standard Score', "
            "'Age-Corrected Standard Score', 'National Percentile (age adjusted)', 'Fully-Corrected T-score' as columns."
        )
    )
    parser.add_argument(
        "--unorganized_nih_toolbox_file",
        dest="unorganized_nih_toolbox_file",
        required=True,
        help="Absolute path of to the unorganized NIH toolbox data.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        default=None,
        required=False,
        help=(
            "Path to the output directory for the organized NIH toolbox data. "
            "If None, saves to the same directory as input data"
        ),
    )
    parser.add_argument(
        "--prefix_filename",
        dest="prefix_filename",
        default=None,
        required=False,
        help="A prefix to add to the filename for the organized NIH toolbox data.",
    )
    parser.add_argument(
        "--include_assessment_dates",
        dest="include_assessment_dates",
        default=False,
        required=False,
        help="Include the assessment dates from the data.",
    )
    parser.add_argument(
        "--preexisting_nih_toolbox_file",
        dest="preexisting_nih_toolbox_file",
        default=None,
        required=False,
        help="A pre-existing organized NIH toolbox file to append the new data too.",
    )

    return parser


if __name__ == "__main__":
    option = _get_parser().parse_args()
    kwargs = vars(option)
    run_pipeline(**kwargs)
