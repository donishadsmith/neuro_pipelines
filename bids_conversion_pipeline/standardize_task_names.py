import re
from pathlib import Path
from typing import Literal

from nifti2bids.io import regex_glob, get_nifti_header
from nifti2bids.metadata import is_3d_img, infer_task_from_image
from nifti2bids.logging import setup_logger

LGR = setup_logger(__name__)

_TASK_NAMES_PATTERNS = {
    "kids": "(mtl(_neu)?|n([_-])?back|princess|flanker|mprage(32)?)",
    "adults": "(mtl|n([_-])?back|(simple.*(repeat.*)?)?gng|flanker|mprage(32)?)",
}
_TASK_VOLUME_MAP = {
    "kids": {"flanker": 305, "nback": 246, "princess": 262, "mtl": 96},
    "adults": {"flanker": 305, "nback": 124, "mtl": 174, "gng": 98},
}


def _pattern_found(pattern: str, filename: str):
    match = re.search(pattern, filename.lower())

    return True if match else False


def _infer_file_identity(temp_dir: Path, cohort: Literal["kids", "adults"]) -> None:
    """
    For files with no task names, identify the file by whether its
    3D (anatomical) or not. If it is a 4D image, then infer the
    task by the number of volumes.
    """
    nifti_files = regex_glob(temp_dir, pattern=r"^.*\.nii\.gz$", recursive=True)
    for nifti_file in nifti_files:
        if not _pattern_found(_TASK_NAMES_PATTERNS[cohort], nifti_file.name):
            if is_3d_img(nifti_file):
                # Safety identity check based on voxel sizes for near isotropic mprage32
                # Note: Protocol doesn't collect fmaps
                min_thresh = 0.8
                max_thresh = 1.1
                voxel_sizes = get_nifti_header(nifti_file).get_zooms()
                if all(
                    min_thresh <= vox_size <= max_thresh for vox_size in voxel_sizes
                ):
                    desc = "mprage32"
                else:
                    LGR.critical(
                        f"Voxel sizes out of bounds ({min_thresh} mm, {max_thresh} mm). "
                        f"Actual voxels sizes are {voxel_sizes} for the following file: {nifti_file}. "
                        "Check original file in the source directory since the temp file will be deleted "
                        "to allow the conversion to continue for the remaining files."
                    )
                    nifti_file.unlink()
                    continue
            else:
                desc = infer_task_from_image(nifti_file, _TASK_VOLUME_MAP[cohort])

            # Special case since there are two mtl files for the kids dataset and the adult dataset
            # with the same number of volumes
            # Each mtl file has the acquisition number in the filename that preceeds "_1"
            # MTLE comes before the MTLR hence the acquisition number is important
            # Same for go-nogo, there are two versions with the same number of volumes for the adult data
            if desc.startswith("mtl") or desc.endswith("gng"):
                # There are cases where there is only a single number followed by
                # extension hence the _\d+ is optional
                pattern = r"_(\d+)(?:_\d+)?\.nii\.gz$"
                # Can be _{acquisition_number}_1.nii.gz or _{acquisition_number}.nii.gz
                match_str = re.search(pattern, nifti_file.name).group(0)
                desc += match_str

                # Get the name before the acquisition number
                prefix_filename = str(nifti_file).split(match_str)[0]
                new_filename = f"{prefix_filename}_{desc}"
            else:
                new_filename = f"{str(nifti_file).split('.nii.gz')[0]}_{desc}.nii.gz"

            nifti_file.rename(new_filename)


def _standardize_task_pipeline(
    temp_dir: Path, cohort: Literal["kids", "adults"]
) -> None:
    _infer_file_identity(temp_dir, cohort)
    _differentiate_mtl_filenames(temp_dir)
    _standardize_nback_filenames(temp_dir)

    if cohort == "adults":
        _differentiate_gng_filenames(temp_dir)


def _differentiate_filenames(
    temp_dir: Path, task: str, task_order_dict: dict[int, str]
):
    for subject_folder in temp_dir.glob("*"):
        nifti_files = list(
            regex_glob(subject_folder, pattern=r"^.*\.nii\.gz$", recursive=True)
        )
        nifti_files = [
            nifti_file
            for nifti_file in nifti_files
            if _pattern_found(_get_pattern(task), nifti_file.name)
        ]

        # Cant sort due to lexicographical sorting which makes 10 preceed 9.
        # Solution: create a list of tuples where the first element is the acquisition
        # number extracted using regex and the second element is the path
        # Then the sorting of tuples is done on the acquisition numbers

        # Note an alternative could be sorting based on modified or created time;
        # however; this may not be as reliable on Unix vs Windows

        # There are cases where there is only a single number followed by
        # extension hence the _\d+ is optional
        pattern = r"_(\d+)(?:_\d+)?\.nii\.gz$"
        nii_tuple_list = sorted(
            [
                (int(re.search(pattern, str(nifti_file)).group(1)), nifti_file)
                for nifti_file in nifti_files
            ]
        )

        for index, nii_tuple in enumerate(nii_tuple_list):
            _, nifti_file = nii_tuple
            task_name = task_order_dict[index]
            new_nifti_filename = nifti_file.parent / nifti_file.name.lower().replace(
                _get_replace_name(nifti_file, task), task_name
            )
            nifti_file.rename(new_nifti_filename)


def _get_pattern(task: Literal["mtl", "gng", "nback"]):
    return {
        "mtl": "(mtl(_neu)?)",
        "gng": "((simple.*(repeat.*)?)?gng)",
        "nback": "(n([_-])?back)",
    }[task]


def _get_replace_name(nifti_file: Path, task: Literal["mtl", "gng", "nback"]):
    return re.search(_get_pattern(task), nifti_file.name.lower()).group(1)


def _differentiate_mtl_filenames(temp_dir: Path) -> None:
    _differentiate_filenames(
        temp_dir, task="mtl", task_order_dict={0: "mtle", 1: "mtlr"}
    )


def _differentiate_gng_filenames(temp_dir: Path) -> None:
    _differentiate_filenames(
        temp_dir, task="gng", task_order_dict={0: "simplegng", 1: "complexgng"}
    )


def _standardize_nback_filenames(temp_dir: Path) -> None:
    nifti_files = list(regex_glob(temp_dir, pattern=r"^.*\.nii\.gz$", recursive=True))
    nifti_files = [
        nifti_file
        for nifti_file in nifti_files
        if _pattern_found(_get_pattern("nback"), nifti_file.name)
    ]
    for nifti_file in nifti_files:
        new_filename = nifti_file.parent / nifti_file.name.lower().replace(
            _get_replace_name(nifti_file, "nback"), "nback"
        )
        nifti_file.rename(new_filename)
