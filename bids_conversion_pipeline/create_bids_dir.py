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

from _utils import (
    _get_constant,
    _create_or_append_participants_tsv,
    _extract_subjects_visits_data,
    _standardize_dates,
)

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


def _get_folder_scan_dates(subject_nifti_files: list[Path]) -> list[str]:
    return sorted(
        list(
            set(
                [
                    subject_nifti_files.parent.name.split("_")[-1]
                    for subject_nifti_files in subject_nifti_files
                ]
            )
        )
    )


def _get_subject_visits(
    subject_id: str,
    subjects_visits_df: pd.DataFrame,
    subjects_visits_date_fmt: str,
    src_data_date_fmt: str,
) -> dict[str, str]:
    # Don't sort to keep the order of the NaNs
    visit_dates = _extract_subjects_visits_data(
        subject_id, subjects_visits_df, column_name="date"
    )

    if not visit_dates or all(not pd.notna(date) for date in visit_dates):
        LGR.critical(f"Subject {subject_id} has no visit dates.")

        return None

    check_dates = [date for date in visit_dates if not pd.notna(date)]
    if not all(
        is_valid_date(visit_date, subjects_visits_date_fmt)
        for visit_date in check_dates
    ):
        LGR.critical(
            f"Visit dates will be ignored for subject {subject_id} because "
            f"not all dates have a consistent format: {check_dates}."
        )

        return None

    visit_dates = _standardize_dates(visit_dates, subjects_visits_date_fmt)

    convert_date = lambda date: (
        datetime.strptime(date, subjects_visits_date_fmt).strftime(src_data_date_fmt)
        if isinstance(date, str)
        else float("NaN")
    )

    visit_dates = [
        str(date) if not isinstance(date, float) else date for date in visit_dates
    ]
    return {
        f"0{session_id}": date
        for session_id, date in enumerate(list(map(convert_date, visit_dates)), start=1)
    }


def _get_subject_dosages(
    subject_id: str,
    subjects_visits_df: pd.DataFrame,
) -> dict[str, str] | None:
    dosages = (
        _extract_subjects_visits_data(
            subject_id, subjects_visits_df, column_name="dose"
        )
        if "dose" in subjects_visits_df.columns
        else None
    )
    if dosages is None:
        return None
    else:
        return {
            f"0{session_id}": dosage
            for session_id, dosage in enumerate(dosages, start=1)
        }


def _combine_session_data(
    visit_session_map: dict[str, str] | None,
    scan_dates: list[str],
    visit_dosage_map: dict[str, str] | None,
) -> list[tuple[str, str, float]]:
    session_scan_date_map = {}
    if visit_session_map:
        session_scan_date_map = {
            session_id: date
            for session_id, date in visit_session_map.items()
            if date in scan_dates
        }

    if not session_scan_date_map:
        session_scan_date_map = {
            f"0{session_id}": date
            for session_id, date in enumerate(scan_dates, start=1)
        }

    filtered_dosages = []
    if visit_dosage_map:
        filtered_dosages = [
            float(dosage)
            for session_id, dosage in visit_dosage_map.items()
            if session_id in session_scan_date_map
        ]

    if not filtered_dosages:
        filtered_dosages = [float("NaN")] * len(session_scan_date_map)

    return zip(
        session_scan_date_map.keys(), session_scan_date_map.values(), filtered_dosages
    )


def _create_sessions_tsv(
    bids_dir: Path, subject_id: str, sessions_dict: dict[str, str]
) -> None:
    new_sessions_df = pd.DataFrame(sessions_dict)
    new_sessions_df["session_id"] = [
        f"ses-{session_id}" if not session_id.startswith("ses-") else session_id
        for session_id in new_sessions_df["session_id"].to_numpy(copy=True)
    ]
    filename = bids_dir / f"sub-{subject_id}" / f"sub-{subject_id}_sessions.tsv"
    new_sessions_df.to_csv(filename, index=False, sep="\t")


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
        scan_dates = _get_folder_scan_dates(subject_nifti_files)
        if scan_dates:
            scan_dates = _standardize_dates(scan_dates, src_data_date_fmt)

        if not all(is_valid_date(date, src_data_date_fmt) for date in scan_dates):
            LGR.warning(
                f"Not all dates have the following format ({src_data_date_fmt}) "
                f"for subject {subject_id}: {scan_dates}."
            )

        if subjects_visits_df is not None:
            visit_session_map = _get_subject_visits(
                subject_id,
                subjects_visits_df,
                subjects_visits_date_fmt,
                src_data_date_fmt,
            )
            visit_dosage_map = (
                _get_subject_dosages(subject_id, subjects_visits_df)
                if dataset == "mph"
                else None
            )
        else:
            visit_session_map, visit_dosage_map = None, None

        session_data_tuple = _combine_session_data(
            visit_session_map, scan_dates, visit_dosage_map
        )

        sessions_dict = {"session_id": [], "acq_time": [], "dose": []}
        for session_id, scan_date, dose in session_data_tuple:
            # Max three sessions
            session_nifti_file = _filter_nifti_files(subject_nifti_files, scan_date)[0]
            sessions_dict["session_id"].append(session_id)
            sessions_dict["acq_time"].append(scan_date)
            sessions_dict["dose"].append(dose)

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
                session_nifti_file,
                dst_path,
                subject_id,
                session_id,
                task_id,
                delete_temp_dir,
            )

        if dataset != "mph":
            del sessions_dict["doses"]

        if add_sessions_tsv or subjects_visits_file:
            _create_sessions_tsv(
                bids_dir,
                subject_id,
                sessions_dict,
            )

    if create_dataset_metadata:
        _generate_dataset_metadata(bids_dir, dataset)

    if delete_temp_dir:
        shutil.rmtree(temp_dir)
