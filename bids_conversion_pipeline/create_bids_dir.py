import shutil
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from nifti2bids.io import regex_glob
from nifti2bids.bids import (
    create_bids_file,
    create_dataset_description,
    save_dataset_description,
)
from _utils import _get_constant, _create_or_append_participants_tsv

_TASK_NAMES = {
    "mph": {"kids": ["mtlr", "mtle", "nback", "princess", "flanker"], "adults": None},
    "naag": None,
}


def _filter_nifti_files(nifti_files: list[Path], target: str) -> list[Path]:
    return sorted(
        [nifti_file for nifti_file in nifti_files if target in nifti_file.parent.name]
    )


def _get_task_name(
    nifti_file: Path, dataset: Literal["mph", "naag"], cohort: Literal["kids", "adults"]
) -> str:
    task_names = _get_constant(_TASK_NAMES, dataset, cohort)
    indx = [task in nifti_file.name.lower() for task in task_names].index(True)

    return task_names[indx]


def _rename_file(
    nifti_file: Path,
    bids_dir: Path,
    subject_id: str,
    session_id: str,
    task_id: Optional[str] = None,
) -> None:
    kwargs = {
        "src_file": nifti_file,
        "dst_dir": bids_dir,
        "sub_id": subject_id,
        "ses_id": session_id,
        "run_id": "01",
        "remove_src_file": True,
    }
    if nifti_file.parent.name == "anat":
        create_bids_file(**kwargs, desc="T1w")
    else:
        create_bids_file(**kwargs, task_id=task_id, desc="bold")


def _create_sessions_tsv(
    bids_dir: Path, sessions_dict: dict[str, str], subject_id: str
) -> None:
    new_sessions_df = pd.DataFrame(sessions_dict)
    filename = bids_dir / f"sub-{subject_id}" / f"sub-{subject_id}_sessions.tsv"
    new_sessions_df.to_csv(filename, index=False, sep="\t")


def _generate_dataset_metadata(bids_dir: Path, dataset: Literal["mph", "naag"]) -> None:
    if not list(bids_dir.glob("dataset_description.json")):
        dataset_description = create_dataset_description(
            dataset.upper(), bids_version="1.10.0"
        )
        save_dataset_description(dataset_description, bids_dir)

    _create_or_append_participants_tsv(bids_dir)


def _generate_bids_dir_pipeline(
    temp_dir: Path,
    bids_dir: Path,
    dataset: Literal["mph", "naag"],
    cohort: Literal["kids", "adults"],
    create_dataset_metadata: bool,
    add_sessions_tsv: bool,
) -> None:
    nifti_files = regex_glob(temp_dir, pattern=r"^.*\.nii\.gz$", recursive=True)

    if not bids_dir.exists():
        bids_dir.mkdir()

    subject_ids = sorted(
        list(set([nifti_file.parent.name.split("_")[0] for nifti_file in nifti_files]))
    )
    for subject_id in subject_ids:
        subject_nifti_files = _filter_nifti_files(nifti_files, subject_id)
        scan_dates = sorted(
            list(
                set(
                    [
                        subject_nifti_files.parent.name.split("_")[-1]
                        for subject_nifti_files in subject_nifti_files
                    ]
                )
            )
        )

        sessions_dict = {"session_id": [], "acq_time": []}
        for session, scan_date in enumerate(scan_dates, start=1):
            session_id = f"0{session}"
            session_nifti_files = _filter_nifti_files(subject_nifti_files, scan_date)
            sessions_dict["session_id"].append(session_id)
            sessions_dict["acq_time"].append(scan_date)
            for session_nifti_file in session_nifti_files:
                dst_path = (
                    bids_dir
                    / f"sub-{subject_id}"
                    / f"ses-{session_id}"
                    / ("anat" if "mprage" in session_nifti_file.name.lower() else "func")
                )

                task_id = (
                    _get_task_name(session_nifti_file, dataset, cohort)
                    if dst_path.name == "func"
                    else None
                )
                _rename_file(
                    session_nifti_file, dst_path, subject_id, session_id, task_id
                )

        if add_sessions_tsv:
            _create_sessions_tsv(bids_dir, sessions_dict, subject_id)

    if create_dataset_metadata:
        _generate_dataset_metadata(bids_dir, dataset)

    shutil.rmtree(temp_dir)
