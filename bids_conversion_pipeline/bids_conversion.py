import tempfile, shutil, sys
from pathlib import Path
from typing import Literal, Optional

from nibabel.filebasedimages import ImageFileError

from bidsaid._decorators import check_nifti
from bidsaid.io import _copy_file, load_nifti, compress_image, regex_glob
from bidsaid.logging import setup_logger
from bidsaid.path_utils import is_valid_date

from standardize_task_names import _standardize_task_pipeline
from create_bids_dir import _generate_bids_dir_pipeline
from create_metadata import _create_json_sidecar_pipeline
from _bids_conversion_utils import _check_subjects_visits_file, _strip_entity

LGR = setup_logger(__name__)


def _resolve_directories(bids_dir, temp_dir):
    if not bids_dir:
        bids_dir = Path().home() / "BIDS_Dataset"
    else:
        bids_dir = Path(bids_dir)

    if not bids_dir.exists():
        bids_dir.mkdir()

    use_tempfile = bool(temp_dir)
    temp_dir = temp_dir or tempfile.TemporaryDirectory().name
    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        temp_dir.mkdir()
    elif temp_dir.exists() and not use_tempfile:
        raise FileExistsError(
            "The temporary directory exists; either choose another name or "
            f"delete the following directory: {temp_dir}"
        )

    return bids_dir, temp_dir


def _filter_subjects(
    folders: list[Path], subjects: Optional[list[str | int]]
) -> list[Path]:
    if subjects:
        return [folder for folder in folders if folder.name.split("_")[0] in subjects]
    else:
        return folders


def _filter_source_folders(
    folders: list[Path], exclude_src_folder_names: Optional[list[str]]
):
    if not exclude_src_folder_names:
        return folders

    exclude_src_folder_names = [
        Path(folder).name for folder in exclude_src_folder_names
    ]

    return [folder for folder in folders if folder.name not in exclude_src_folder_names]


def _copy_nifti_files(nifti_file: Path, temp_dir: Path) -> None:
    dst_file = temp_dir / nifti_file.parent.name / nifti_file.name
    _copy_file(
        src_file=nifti_file,
        dst_file=dst_file,
        remove_src_file=False,
    )

    try:
        load_nifti(dst_file)
    except ImageFileError:
        LGR.warning(
            "Deleting the following nifti file from the temporary directory since "
            f"Nibabel cannot work out file type: {nifti_file}",
            exc_info=True,
        )
        dst_file.unlink()
        return

    try:
        _is_raw_nifti(dst_file)
    except ValueError:
        LGR.warning(
            "Deleting the following nifti file from the temporary directory "
            f"since it is likely not a raw image: {nifti_file}",
            exc_info=True,
        )
        dst_file.unlink()
        return

    participant_id = nifti_file.parent.name.split("_")[0]
    if nifti_file.name.split("_")[0] != participant_id:
        LGR.warning(
            "Deleting the following nifti file from the temporary directory "
            "since it is nested in a source directory with a different subject "
            f"id ({participant_id}): {nifti_file}"
        )
        dst_file.unlink()
        return

    if nifti_file.name.endswith(".nii"):
        try:
            compress_image(dst_file, dst_file.parent, remove_src_file=True)
        except OSError:
            LGR.warning(
                f"An OSError occured while compressing the following file: {dst_file}. "
                "Removing file from the temporary directory."
            )
            dst_file.unlink()


@check_nifti(nifti_param_name="nifti_file")
def _is_raw_nifti(nifti_file):
    pass


def _copy_data_to_temp_dir(
    src_dir: Path,
    temp_dir: Path,
    subjects: Optional[list[str | int]],
    exclude_src_folder_names: Optional[list[str]],
    exclude_nifti_filenames: Optional[list[str]],
) -> None:
    subject_folders = regex_glob(src_dir, pattern=r"^\d+.*_\d+$")
    subject_folders = _filter_subjects(subject_folders, subjects)
    subject_folders = _filter_source_folders(subject_folders, exclude_src_folder_names)
    if not subject_folders:
        LGR.warning(
            f"After filtering, no folders can be converted from the following directory: {src_dir}"
        )
        sys.exit(1)

    exclude_nifti_filenames = exclude_nifti_filenames or []
    exclude_nifti_filenames = [Path(file).name for file in exclude_nifti_filenames]
    for subject_folder in subject_folders:
        date_str = subject_folder.name.split("_")[-1]
        if not is_valid_date(date_str, "%y%m%d") or len(date_str) != 6:
            LGR.warning(
                f"The following folder does not have the '%y%m%d' date format: {subject_folder}. "
                "Dates are sorted and should be standardized across all folders."
            )

        # Handle edge case where analyses related files placed in same folder as
        # original nifti which has the naming {subjectID}_{scan_date}_{acqusition_number}
        nifti_files = regex_glob(
            subject_folder, pattern=r"^\d+_(\d+)?.*\.(nii|nii.gz)$"
        )
        for nifti_file in nifti_files:
            if nifti_file.name in exclude_nifti_filenames:
                continue

            _copy_nifti_files(nifti_file, temp_dir)


def run_pipeline(
    src_dir: str,
    temp_dir: str,
    bids_dir: str,
    subjects: Optional[list[str | int]],
    exclude_src_folder_names: Optional[list[str]],
    exclude_nifti_filenames: Optional[list[str]],
    cohort: Literal["kids", "adults"],
    delete_temp_dir: bool,
    create_dataset_metadata: bool,
    add_sessions_tsv: bool,
    subjects_visits_file: str,
    subjects_visits_date_fmt: str,
    src_data_date_fmt: str,
) -> Path:
    try:
        if (cohort := cohort.lower()) not in ["kids", "adults"]:
            raise ValueError("'--cohort' must be 'kids' or 'adults'.")

        _check_subjects_visits_file(subjects_visits_file, dose_column_required=False)

        bids_dir, temp_dir = _resolve_directories(bids_dir, temp_dir)

        if subjects:
            subjects = _strip_entity(subjects)

        _copy_data_to_temp_dir(
            Path(src_dir),
            temp_dir,
            subjects,
            exclude_src_folder_names,
            exclude_nifti_filenames,
        )

        # Pipeline to identify un-named NIfTI images and standardize task names
        _standardize_task_pipeline(temp_dir, cohort)

        # Pipeline to move files to BIDS directory
        _generate_bids_dir_pipeline(
            temp_dir,
            bids_dir,
            cohort,
            create_dataset_metadata,
            add_sessions_tsv,
            delete_temp_dir,
            subjects_visits_file,
            subjects_visits_date_fmt,
            src_data_date_fmt,
        )

        # Pipeline to create JSON sidecars for NIfTI images
        _create_json_sidecar_pipeline(bids_dir, cohort)
    finally:
        if delete_temp_dir and (isinstance(temp_dir, Path) and temp_dir.exists()):
            shutil.rmtree(temp_dir)

    return bids_dir
