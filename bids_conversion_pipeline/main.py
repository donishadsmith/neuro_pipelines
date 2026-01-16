import argparse, tempfile, shutil
from pathlib import Path
from typing import Literal, Optional

from nifti2bids.io import _copy_file, compress_image, regex_glob
from nifti2bids.logging import setup_logger
from nifti2bids.metadata import is_valid_date

from standardize_task_names import _standardize_task_pipeline
from create_bids_dir import _generate_bids_dir_pipeline
from create_metadata import _create_json_sidecar_pipeline
from _utils import _check_subjects_visits_file, _strip_entity

LGR = setup_logger(__name__)


def _get_cmd_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline to convert dataset to BIDS.")
    parser.add_argument(
        "--src_dir",
        dest="src_dir",
        required=True,
        help=(
            "Source directory containing original data where NIfTI files "
            "are stored in folders with the following format {subject_ID}_{date}."
        ),
    )
    parser.add_argument(
        "--temp_dir",
        dest="temp_dir",
        required=False,
        default=None,
        help="Temporary directory to store intermediate content in.",
    )
    parser.add_argument(
        "--bids_dir", dest="bids_dir", required=True, help="The BIDS directory."
    )
    parser.add_argument(
        "--subjects",
        dest="subjects",
        required=False,
        nargs="+",
        default=None,
        help="The subject IDs in the 'src_dir' to convert to BIDS.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        required=False,
        default="mph",
        help="Name of the dataset (i.e., mph and naag).",
    )
    parser.add_argument(
        "--delete_temp_dir",
        dest="delete_temp_dir",
        required=False,
        default=True,
        type=_convert_to_bool,
        help="Deletes the temporary directory.",
    )
    parser.add_argument(
        "--cohort",
        dest="cohort",
        required=False,
        default="kids",
        help="The cohort if dataset is 'mph' (i.e., kids and adult).",
    )
    parser.add_argument(
        "--create_dataset_metadata",
        dest="create_dataset_metadata",
        required=False,
        default=False,
        type=_convert_to_bool,
        help=(
            "Creates the participant TSV and the dataset description JSON. "
            "If a TSV file is already present in ``bids_dir``, appends the new subject IDs to it. "
            "Also skips the dataset description JSON if detected in ``bids_dir``."
            "**Keep false if running pipeline in parallel to prevent race condition issues.**"
        ),
    )
    parser.add_argument(
        "--add_sessions_tsv",
        dest="add_sessions_tsv",
        required=False,
        default=False,
        type=_convert_to_bool,
        help=(
            "Add basic sessions TSV file containing the session "
            "and scan date in BIDS folder for each subject. "
        ),
    )
    # Extracting the file creation or modification date may not be very reliable
    parser.add_argument(
        "--subjects_visits_file",
        dest="subjects_visits_file",
        required=True,
        help=(
            "A text file, where the 'subject_id' contaims the subject ID and the "
            "'date' column is the date of visit. Using this parameter is recommended "
            "when data is missing. Ensure all dates have a consistent format. "
            "**All subject visit dates should be listed.** If a 'dose' column is included, "
            "then dosages will be included in the sessions TSV file."
        ),
    )
    parser.add_argument(
        "--subjects_visits_date_fmt",
        dest="subjects_visits_date_fmt",
        required=False,
        default=r"%m/%d/%Y",
        help=("The format of the date in the ``subjects_visits_file`` file."),
    )
    parser.add_argument(
        "--src_data_date_fmt",
        dest="src_data_date_fmt",
        required=False,
        default=r"%y%m%d",
        help=(
            "The format of the dates in the filenames that are in the source directory."
        ),
    )

    return parser


def _convert_to_bool(arg: bool | str) -> bool:
    if str(arg).lower() == "true":
        return True
    elif str(arg).lower() == "false":
        return False
    else:
        raise ValueError("For booleans only True and False are valid.")


def _filter_subjects(
    folders: list[Path], subjects: Optional[list[str | int]]
) -> list[Path]:
    if subjects:
        return [folder for folder in folders if folder.name.split("_")[0] in subjects]
    else:
        return folders


def _copy_nifti_files(nifti_file: Path, temp_dir: Path) -> None:
    dst_file = temp_dir / nifti_file.parent.name / nifti_file.name
    _copy_file(
        src_file=nifti_file,
        dst_file=dst_file,
        remove_src_file=False,
    )

    if nifti_file.name.endswith(".nii"):
        try:
            compress_image(dst_file, dst_file.parent, remove_src_file=True)
        except OSError:
            LGR.critical(
                f"An OSError occured while compressing the following file: {dst_file}. "
                "Removing file from the temporary directory."
            )
            dst_file.unlink()


def _copy_data_to_temp_dir(
    src_dir: Path, temp_dir: Path, subjects: Optional[list[str | int]]
) -> None:
    subject_folders = _filter_subjects(
        folders=regex_glob(src_dir, pattern=r"^\d+_\d+$"), subjects=subjects
    )

    for subject_folder in subject_folders:
        date_str = subject_folder.name.split("_")[-1]
        if not is_valid_date(date_str, "%y%m%d") or len(date_str) != 6:
            LGR.critical(
                f"The following folder does not have the '%y%m%d' date format: {subject_folder}. "
                "Dates are sorted and should be standardized across all folders."
            )

        # Handle edge case where analyses related files placed in same folder as
        # original nifti which has the naming {subjectID}_{scan_date}_{acqusition_number}
        nifti_files = regex_glob(subject_folder, pattern=r"^\d+_\d+.*\.(nii|nii.gz)$")
        for nifti_file in nifti_files:
            _copy_nifti_files(nifti_file, temp_dir)


def main(
    src_dir: str,
    temp_dir: str,
    bids_dir: str,
    subjects: Optional[list[str | int]],
    dataset: Literal["mph", "naag"],
    cohort: Literal["kids", "adults"],
    delete_temp_dir: bool,
    create_dataset_metadata: bool,
    add_sessions_tsv: bool,
    subjects_visits_file: str,
    subjects_visits_date_fmt: str,
    src_data_date_fmt: str,
) -> None:
    try:
        if (dataset := dataset.lower()) not in ["naag", "mph"]:
            raise ValueError("'--dataset' must be 'naag' or 'mph'.")

        if (cohort := cohort.lower()) not in ["kids", "adults"]:
            raise ValueError("'--cohort' must be 'kids' or 'adults'.")

        if subjects_visits_file:
            _check_subjects_visits_file(
                subjects_visits_file, dose_column_required=False
            )

        # Create temporary directory with compressed files
        temp_dir = temp_dir or tempfile.TemporaryDirectory().name
        temp_dir: Path = Path(temp_dir)
        if not temp_dir.exists():
            temp_dir.mkdir()

        bids_dir = Path(bids_dir)

        if subjects:
            subjects = _strip_entity(subjects)

        _copy_data_to_temp_dir(Path(src_dir), temp_dir, subjects)

        # Pipeline to identify un-named NIfTI images and standardize task names
        _standardize_task_pipeline(temp_dir, dataset, cohort)

        # Pipeline to move files to BIDS directory
        _generate_bids_dir_pipeline(
            temp_dir,
            bids_dir,
            dataset,
            cohort,
            create_dataset_metadata,
            add_sessions_tsv,
            delete_temp_dir,
            subjects_visits_file,
            subjects_visits_date_fmt,
            src_data_date_fmt,
        )

        # Pipeline to create JSON sidecars for NIfTI images
        _create_json_sidecar_pipeline(bids_dir)
    finally:
        if delete_temp_dir and (isinstance(temp_dir, Path) and temp_dir.exists()):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    args = _get_cmd_args().parse_args()
    main(**vars(args))
