import re
from pathlib import Path
from typing import Literal

from nifti2bids.io import regex_glob, get_nifti_header
from nifti2bids.metadata import is_3d_img, infer_task_from_image
from nifti2bids.logging import setup_logger

from _utils import _get_constant

_TASK_NAMES = {
    "mph": {
        "kids": ["mtl_neu", "n-back", "nback", "n_back", "princess", "flanker"],
        "adults": None,
    },
    "naag": None,
}
_ANAT_NAME = "mprage32"
_TASK_VOLUME_MAP = {
    "mph": {
        "kids": {"flanker": 305, "nback": 246, "princess": 262, "mtl": 96},
        "adults": None,
    },
    "naag": None,
}

LGR = setup_logger(__name__)


def _infer_file_identity(
    temp_dir: Path, all_desc: list[str], task_volume_map: dict[str, int]
) -> None:
    """
    For files with no task names, identify the file by whether its
    3D (anatomical) or not. If it is a 4D image, then infer the
    task by the number of volumes.
    """
    nifti_files = regex_glob(temp_dir, pattern=r"^.*\.nii\.gz$", recursive=True)
    for nifti_file in nifti_files:
        if not any(name in nifti_file.name.lower() for name in all_desc):
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
                desc = infer_task_from_image(nifti_file, task_volume_map)

            # Special case since there are two mtl_neu with 96 volumes
            # Each mtl file has the acquisition number in the filename that preceeds "_1"
            # MTLE comes before the MTLR hence the acquisition number is important
            if desc.startswith("mtl"):
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
    temp_dir: Path, dataset: Literal["mph", "naag"], cohort: Literal["kids", "adults"]
) -> None:
    all_desc = _get_constant(_TASK_NAMES, dataset, cohort) + [_ANAT_NAME]
    task_volume_map = _get_constant(_TASK_VOLUME_MAP, dataset, cohort)

    _infer_file_identity(temp_dir, all_desc, task_volume_map)

    if dataset == "mph":
        _rename_mtl_filenames(temp_dir)
        _standardize_nback_filenames(temp_dir, all_desc)


def _rename_mtl_filenames(temp_dir: Path) -> None:
    for subject_folder in temp_dir.glob("*"):
        nifti_files = list(
            regex_glob(subject_folder, pattern=r"^.*\.nii\.gz$", recursive=True)
        )
        nifti_files = [
            nifti_file for nifti_file in nifti_files if "mtl" in nifti_file.name.lower()
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

        for indx, nii_tuple in enumerate(nii_tuple_list):
            _, nifti_file = nii_tuple
            task_name = "mtle" if indx == 0 else "mtlr"
            replace_name = "mtl_neu" if "mtl_neu" in nifti_file.name else "mtl"
            new_nifti_filename = nifti_file.parent / nifti_file.name.lower().replace(
                replace_name, task_name
            )
            nifti_file.rename(new_nifti_filename)


def _standardize_nback_filenames(temp_dir: Path, all_desc: list[str]) -> None:
    nback_variants = [desc for desc in all_desc if desc.endswith("back")]
    nifti_files = list(regex_glob(temp_dir, pattern=r"^.*\.nii\.gz$", recursive=True))

    nifti_files = [
        nifti_file
        for nifti_file in nifti_files
        if any(variant in nifti_file.name.lower() for variant in nback_variants)
    ]
    for nifti_file in nifti_files:
        indx = [variant in nifti_file.name.lower() for variant in nback_variants].index(
            True
        )
        variant = nback_variants[indx]
        if variant == "nback":
            continue

        new_filename = nifti_file.parent / nifti_file.name.lower().replace(
            variant, "nback"
        )

        nifti_file.rename(new_filename)
