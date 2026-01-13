import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np, pandas as pd

from nifti2bids.io import regex_glob
from nifti2bids.bids import (
    create_bids_file,
    create_dataset_description,
    save_dataset_description,
)
from nifti2bids.metadata import is_valid_date
from nifti2bids.logging import setup_logger

from _utils import _get_constant, _create_or_append_participants_tsv

LGR = setup_logger(__name__)

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
    remove_src_file: bool = True,
) -> None:
    kwargs = {
        "src_file": nifti_file,
        "dst_dir": bids_dir,
        "sub_id": subject_id,
        "ses_id": session_id,
        "run_id": "01",
        "remove_src_file": remove_src_file,
    }

    if bids_dir.name == "anat":
        create_bids_file(**kwargs, desc="T1w")
    else:
        create_bids_file(**kwargs, task_id=task_id, desc="bold")


def _create_sessions_tsv(
    bids_dir: Path, sessions_dict: dict[str, str], subject_id: str
) -> None:
    new_sessions_df = pd.DataFrame(sessions_dict)
    new_sessions_df["session_id"] = [
        f"ses-{session_id}" if not session_id.startswith("ses-") else session_id
        for session_id in new_sessions_df["session_id"].values
    ]
    filename = bids_dir / f"sub-{subject_id}" / f"sub-{subject_id}_sessions.tsv"
    new_sessions_df.to_csv(filename, index=False, sep="\t")


def _generate_dataset_metadata(bids_dir: Path, dataset: Literal["mph", "naag"]) -> None:
    if not list(bids_dir.glob("dataset_description.json")):
        dataset_description = create_dataset_description(
            dataset.upper(), bids_version="1.4.0"
        )
        save_dataset_description(dataset_description, bids_dir)

    _create_or_append_participants_tsv(bids_dir)


def _get_dataframe(subjects_visits_file: str | Path) -> pd.DataFrame | None:
    if not subjects_visits_file:
        return None

    return pd.read_csv(subjects_visits_file, sep=None, engine="python")


def _get_subjects_visits(
    subject_id: str,
    subjects_visits_df: pd.DataFrame,
    subjects_visits_date_fmt: str,
    src_data_date_fmt: str,
) -> dict[str, str]:
    # Don't sort to keep the order of the NaNs
    visit_dates = (
        subjects_visits_df[subjects_visits_df.iloc[:, 0].astype(str) == subject_id]
        .iloc[:, 1]
        .values.tolist()
    )

    if not visit_dates or all(
        isinstance(date, float) and np.isnan(date) for date in visit_dates
    ):
        LGR.critical(f"Subject {subject_id} has no visit dates.")

        return None

    check_dates = [
        date for date in visit_dates if not (isinstance(date, float) and np.isnan(date))
    ]
    if not all(
        is_valid_date(visit_date, subjects_visits_date_fmt)
        for visit_date in check_dates
    ):
        LGR.critical(
            f"Visit dates will be ignored for subject {subject_id} because not all dates have a consistent format: "
            f"{check_dates}."
        )

        return None

    # Format of the event files are hardcoded into the presentation script
    convert_date = lambda date: datetime.strptime(
        date, subjects_visits_date_fmt
    ).strftime(src_data_date_fmt)

    return {
        f"0{session_id}": date
        for session_id, date in enumerate(list(map(convert_date, visit_dates)), start=1)
    }


def _generate_bids_dir_pipeline(
    temp_dir: Path,
    bids_dir: Path,
    dataset: Literal["mph", "naag"],
    cohort: Literal["kids", "adults"],
    create_dataset_metadata: bool,
    add_sessions_tsv: bool,
    delete_temp_dir: bool,
    subjects_visits_file: str,
    subjects_visits_date_fmt: str,
    src_data_date_fmt: str,
) -> None:
    nifti_files = regex_glob(temp_dir, pattern=r"^.*\.nii\.gz$", recursive=True)

    if not bids_dir.exists():
        bids_dir.mkdir()

    subjects_visits_df = _get_dataframe(subjects_visits_file)

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

        if not all(is_valid_date(date, src_data_date_fmt) for date in scan_dates):
            LGR.warning(f"Not all dates have the following format ({src_data_date_fmt}) "
                        f"for subject {subject_id}: {scan_dates}.")

        visit_session_map = (
            _get_subjects_visits(
                subject_id,
                subjects_visits_df,
                subjects_visits_date_fmt,
                src_data_date_fmt,
            )
            if subjects_visits_df is not None
            else None
        )
        if visit_session_map:
            session_scan_date_map = {
                session_id: date
                for session_id, date in visit_session_map.items()
                if date in scan_dates
            }
        else:
            session_scan_date_map = {
                f"0{session_id}": date
                for session_id, date in enumerate(scan_dates, start=1)
            }

        sessions_dict = {"session_id": [], "acq_time": []}
        for session_id, scan_date in session_scan_date_map.items():
            # Max three sessions
            session_nifti_files = _filter_nifti_files(subject_nifti_files, scan_date)
            sessions_dict["session_id"].append(session_id)
            sessions_dict["acq_time"].append(scan_date)
            for session_nifti_file in session_nifti_files:
                dst_path = (
                    bids_dir
                    / f"sub-{subject_id}"
                    / f"ses-{session_id}"
                    / (
                        "anat"
                        if "mprage" in session_nifti_file.name.lower()
                        else "func"
                    )
                )

                task_id = (
                    _get_task_name(session_nifti_file, dataset, cohort)
                    if dst_path.name == "func"
                    else None
                )
                _rename_file(
                    session_nifti_file,
                    dst_path,
                    subject_id,
                    session_id,
                    task_id,
                    delete_temp_dir,
                )

        if add_sessions_tsv:
            _create_sessions_tsv(bids_dir, sessions_dict, subject_id)

    if create_dataset_metadata:
        _generate_dataset_metadata(bids_dir, dataset)

    if delete_temp_dir:
        shutil.rmtree(temp_dir)
