import re, shutil, sys
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from bidsaid.io import regex_glob
from bidsaid.files import (
    create_bids_file,
    create_dataset_description,
    get_entity_value,
    save_dataset_description,
)
from bidsaid.logging import setup_logger
from bidsaid.path_utils import is_valid_date

from _bids_conversion_utils import (
    _create_or_append_participants_tsv,
)

from _general_utils import (
    _extract_subjects_visits_data,
    _get_subject_visits,
    _standardize_dates,
)

LGR = setup_logger(__name__)

_TASK_NAMES = {
    "kids": "(mtlr|mtle|nback|princess|flanker)",
    "adults": "(mtlr|mtle|nback|flanker|simplegng|complexgng)",
}


def cross_validate_folders(bids_dir: Path, temp_dir: Path):
    modalities = ["anat", "func"]
    for subject_folder in bids_dir.glob("*"):
        if not subject_folder.is_dir():
            continue

        sub_id = get_entity_value(subject_folder, "sub")
        session_folders = [
            content for content in subject_folder.glob("*") if content.is_dir()
        ]
        for session_folder in session_folders:
            ses_id = get_entity_value(session_folder, "ses")
            modalitiy_folders = session_folder.glob("*")
            missing_modalities = [
                x.name for x in modalitiy_folders if x.name not in modalities
            ]
            if missing_modalities:
                LGR.warning(
                    f"For session {ses_id}, subject {sub_id} is missing the following modality folders: {','.join(missing_modalities)}"
                )

        n_subject_sessions = len(session_folders)
        n_subject_temp_folders = len(list(temp_dir.glob(f"*{sub_id}*")))
        if n_subject_sessions != n_subject_temp_folders:
            LGR.warning(
                f"The number of BIDS sessions ({n_subject_sessions}) for subject {sub_id} does not equal the number of folders "
                f"in the temporary directory containing the subject ID {n_subject_temp_folders}"
            )


def _filter_nifti_files(nifti_files: list[Path], target: str) -> list[Path]:
    return sorted(
        [nifti_file for nifti_file in nifti_files if target in nifti_file.parent.name]
    )


def _get_task_name(nifti_file: Path, cohort: Literal["kids", "adults"]) -> str:
    return re.search(_TASK_NAMES[cohort], nifti_file.name.lower()).group(1)


def _rename_file(
    nifti_file: Path,
    subject_dir: Path,
    participant_id: str,
    session_id: str,
    task_id: Optional[str] = None,
    remove_src_file: bool = True,
) -> None:
    kwargs = {
        "src_file": nifti_file,
        "dst_dir": subject_dir,
        "sub_id": participant_id,
        "ses_id": session_id,
        "run_id": "01",
        "remove_src_file": remove_src_file,
    }

    if subject_dir.name == "anat":
        create_bids_file(**kwargs, desc="T1w")
    else:
        create_bids_file(**kwargs, task_id=task_id, desc="bold")


def _generate_dataset_metadata(bids_dir: Path) -> None:
    if not list(bids_dir.glob("dataset_description.json")):
        dataset_description = create_dataset_description(
            dataset_name="MPH", bids_version="1.4.0"
        )
        save_dataset_description(dataset_description, bids_dir)

    _create_or_append_participants_tsv(bids_dir)


def _get_dataframe(subjects_visits_file: str | Path) -> pd.DataFrame | None:
    if not subjects_visits_file:
        return None

    if str(subjects_visits_file).endswith(".xlsx") or str(
        subjects_visits_file
    ).endswith(".xls"):
        return pd.read_excel(subjects_visits_file)
    else:
        return pd.read_csv(subjects_visits_file, sep=None, engine="python")


def _get_folder_scan_dates(subject_nifti_files: list[Path]) -> list[str]:
    original_scan_dates = sorted(
        list(
            set(
                [
                    subject_nifti_files.parent.name.split("_")[-1]
                    for subject_nifti_files in subject_nifti_files
                ]
            )
        )
    )
    # In the event a folder is missing a date:
    converted_scan_dates = []
    for scan_date, subject_nifti_file in zip(original_scan_dates, subject_nifti_files):
        if not scan_date:
            converted_scan_dates.append("NaN")
        else:
            try:
                # KKI likely has a pipeline that automatically labels folders with the "%y%m%d" date format
                new_dates = [
                    (
                        str(pd.to_datetime([scan_date], format=r"%y%m%d")[0]).split()[0]
                        if is_valid_date(scan_date, r"%y%m%d")
                        else str(pd.to_datetime([scan_date])[0]).split()[0]
                    )
                ]
                converted_scan_dates.append(new_dates[0])
            except:
                LGR.warning(
                    f"The following folder does not have a valid date: {subject_nifti_file.parent}"
                )
                converted_scan_dates.append("NaN")

    original_scan_date_map = {
        k: v for k, v in zip(converted_scan_dates, original_scan_dates)
    }

    return converted_scan_dates, original_scan_date_map


def _get_subject_dosages(
    participant_id: str,
    subjects_visits_df: pd.DataFrame,
) -> dict[str, str] | None:
    dosages = (
        _extract_subjects_visits_data(
            participant_id, subjects_visits_df, column_name="dose"
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
    participant_id: str | int,
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
        missing_dates = {
            date: date
            for _, date in visit_session_map.items()
            if date not in scan_dates
        }
        if missing_dates:
            LGR.warning(
                "The following dates are missing from the subject_visits_file "
                f"for subject {participant_id} and will be used as the session id if a source "
                f"folder has these dates: {missing_dates.keys()}"
            )

            session_scan_date_map.update(missing_dates)

    if not session_scan_date_map:
        session_scan_date_map = {
            f"0{session_id}": date
            for session_id, date in enumerate(scan_dates, start=1)
        }

    filtered_dosages = []
    if visit_dosage_map:
        filtered_dosages = [
            dosage
            for session_id, dosage in visit_dosage_map.items()
            if session_id in session_scan_date_map
        ]

    if not filtered_dosages:
        filtered_dosages = [float("NaN")] * len(session_scan_date_map)

    return zip(
        session_scan_date_map.keys(), session_scan_date_map.values(), filtered_dosages
    )


def _create_sessions_tsv(
    bids_dir: Path, participant_id: str, sessions_dict: dict[str, str]
) -> None:
    new_sessions_df = pd.DataFrame(sessions_dict)
    new_sessions_df["session_id"] = [
        f"ses-{session_id}" if not session_id.startswith("ses-") else session_id
        for session_id in new_sessions_df["session_id"].tolist()
    ]
    filename = bids_dir / f"sub-{participant_id}" / f"sub-{participant_id}_sessions.tsv"
    new_sessions_df.to_csv(filename, index=False, sep="\t")


def _generate_bids_dir_pipeline(
    temp_dir: Path,
    bids_dir: Path,
    cohort: Literal["kids", "adults"],
    create_dataset_metadata: bool,
    add_sessions_tsv: bool,
    delete_temp_dir: bool,
    subjects_visits_file: str,
) -> None:
    nifti_files = list(regex_glob(temp_dir, pattern=r"^.*\.nii\.gz$", recursive=True))

    bids_dir.mkdir(parents=True, exist_ok=True)

    subjects_visits_df = _standardize_dates(_get_dataframe(subjects_visits_file))

    participant_ids = sorted(
        list(set([nifti_file.parent.name.split("_")[0] for nifti_file in nifti_files]))
    )

    for participant_id in participant_ids:
        subject_nifti_files = _filter_nifti_files(nifti_files, participant_id)
        scan_dates, original_scan_date_map = _get_folder_scan_dates(subject_nifti_files)

        visit_session_map = _get_subject_visits(
            participant_id,
            subjects_visits_df,
        )
        visit_dosage_map = _get_subject_dosages(participant_id, subjects_visits_df)

        session_data_tuple = _combine_session_data(
            participant_id, visit_session_map, scan_dates, visit_dosage_map
        )

        sessions_dict = {"session_id": [], "acq_time": [], "dose": []}

        for session_id, converted_scan_date, dose in session_data_tuple:
            sessions_dict["session_id"].append(session_id)
            sessions_dict["acq_time"].append(converted_scan_date)
            sessions_dict["dose"].append(dose)
            session_nifti_files = _filter_nifti_files(
                subject_nifti_files, original_scan_date_map[converted_scan_date]
            )
            for session_nifti_file in session_nifti_files:

                subject_dir = (
                    bids_dir
                    / f"sub-{participant_id}"
                    / f"ses-{session_id}"
                    / (
                        "anat"
                        if "mprage" in session_nifti_file.name.lower()
                        else "func"
                    )
                )

                task_id = (
                    _get_task_name(session_nifti_file, cohort)
                    if subject_dir.name == "func"
                    else None
                )
                _rename_file(
                    session_nifti_file,
                    subject_dir,
                    participant_id,
                    session_id,
                    task_id,
                    delete_temp_dir,
                )

        if add_sessions_tsv or subjects_visits_file:
            _create_sessions_tsv(
                bids_dir,
                participant_id,
                sessions_dict,
            )

    if create_dataset_metadata:
        _generate_dataset_metadata(bids_dir)

    cross_validate_folders(bids_dir, temp_dir)

    if delete_temp_dir:
        shutil.rmtree(temp_dir)
