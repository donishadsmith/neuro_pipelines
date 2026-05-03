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
    _get_dataframe,
    _get_subject_visits,
    _standardize_dates,
)

LGR = setup_logger(__name__)

_TASK_NAMES = {
    "kids": "(mtlr|mtle|nback|princess|flanker)",
    "adults": "(mtlr|mtle|nback|flanker|simplegng|complexgng)",
}


def cross_validate_folders(bids_dir: Path, temp_dir: Path, subjects: list[str]):
    modalities = ["anat", "func"]
    for subject_folder in bids_dir.glob("*"):
        if not subject_folder.is_dir():
            continue

        sub_id = get_entity_value(subject_folder, "sub")
        if subjects and sub_id not in subjects:
            continue

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
                f"in the temporary directory containing the subject ID ({n_subject_temp_folders})"
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


def _get_folder_scan_dates(subject_nifti_files: list[Path]) -> list[str]:
    subject_folders = list(
        set([subject_nifti_file.parent for subject_nifti_file in subject_nifti_files])
    )
    original_scan_dates = sorted(
        [subject_folder.name.split("_")[-1] for subject_folder in subject_folders]
    )

    standardized_scan_dates = []
    for scan_date, subject_folder in zip(original_scan_dates, subject_folders):
        if not scan_date:
            LGR.warning(
                "The following folder does not have a valid date and will "
                f"be removed from the temporary directory: {subject_folder}"
            )
            shutil.rmtree(subject_folder)
            continue

        try:
            # KKI likely has a pipeline that automatically labels folders with the "%y%m%d" date format
            new_dates = [
                (
                    str(pd.to_datetime([scan_date], format=r"%y%m%d")[0]).split()[0]
                    if is_valid_date(scan_date, r"%y%m%d")
                    else str(pd.to_datetime([scan_date])[0]).split()[0]
                )
            ]
            standardized_scan_dates.append(new_dates[0])
        except:
            LGR.warning(
                "The following folder does not have a valid date and will "
                f"be removed from the temporary directory: {subject_folder}"
            )
            shutil.rmtree(subject_folder)

    dates_tuples = list(zip(standardized_scan_dates, original_scan_dates))
    dates_tuples = sorted(dates_tuples, key=lambda x: pd.to_datetime(x[0]))
    standardized_scan_dates, original_scan_dates = zip(*dates_tuples)
    original_scan_date_map = {
        k: v for k, v in zip(standardized_scan_dates, original_scan_dates)
    }

    return standardized_scan_dates, original_scan_date_map


def _get_subject_dosages(
    participant_id: str, subjects_visits_df: pd.DataFrame, scan_dates: list[str]
) -> dict[str, str] | None:
    if "dose" not in subjects_visits_df.columns:
        dosages = None
    else:
        dosages = {
            scan_date: _extract_subjects_visits_data(
                participant_id,
                subjects_visits_df,
                scan_date=scan_date,
                column_name="dose",
            )
            for scan_date in scan_dates
        }
        dosages = {
            scan_date: (dose[0] if dose else float("NaN"))
            for scan_date, dose in dosages.items()
        }

    return dosages


def _combine_session_data(
    participant_id: str | int,
    visit_session_map: dict[str, str] | None,
    scan_dates: list[str],
    visit_dosage_map: dict[str, str] | None,
) -> list[tuple[str, str, float]]:
    # Note, _standardize_dates already sorts on the id and the dates in pandas
    session_scan_date_map = {}
    if visit_session_map:
        session_scan_date_map = {
            session_id: date
            for session_id, date in visit_session_map.items()
            if date in scan_dates
        }
        dates_not_in_source = sorted(
            set(visit_session_map.values()).difference(scan_dates)
        )
        if dates_not_in_source:
            LGR.warning(
                "The following dates are in the subjects visits file but have no corresponding source folder "
                f"for subject {participant_id}: {dates_not_in_source}\n"
                f"Session order based on the subjects visits file will be maintained: {session_scan_date_map}"
            )

        dates_not_in_visits_file = sorted(
            set(scan_dates).difference(visit_session_map.values())
        )
        if dates_not_in_visits_file:
            session_scan_date_map.update(
                {date: date for date in dates_not_in_visits_file}
            )
            LGR.warning(
                "The following dates are in the source folders but not in the subjects visits file "
                f"for subject {participant_id}: {dates_not_in_source}\n"
                f"The date will be used as the session label and the following label mapping will be used: {session_scan_date_map}"
            )

    if not session_scan_date_map:
        # The subject visits file is considered the ultimate authority on session/date ordering
        LGR.warning(
            "Visit session mapping could not be done for subject (participant_id), using all "
            f"dates as session labels {scan_dates}."
        )
        session_scan_date_map = {date: date for date in scan_dates}

    filtered_dosages = []
    if visit_dosage_map:
        reversed_session_scan_date_map = {
            date: ses_id for ses_id, date in session_scan_date_map.items()
        }
        # Going by date order in the session map to guarantee that dosages are always
        # mapped to the correct dates
        filtered_dosages = [
            visit_dosage_map[date] for date in reversed_session_scan_date_map
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
    subjects: list[str],
    create_dataset_metadata: bool,
    add_sessions_tsv: bool,
    delete_temp_dir: bool,
    subjects_visits_file: str,
) -> None:
    nifti_files = list(regex_glob(temp_dir, pattern=r"^.*\.nii\.gz$", recursive=True))

    bids_dir.mkdir(parents=True, exist_ok=True)

    subjects_visits_df = _standardize_dates(
        _get_dataframe(subjects_visits_file), sort_data=True
    )

    participant_ids = sorted(
        list(set([nifti_file.parent.name.split("_")[0] for nifti_file in nifti_files]))
    )

    for participant_id in participant_ids:
        subject_nifti_files = _filter_nifti_files(nifti_files, participant_id)
        standardized_scan_dates, original_scan_date_map = _get_folder_scan_dates(
            subject_nifti_files
        )

        visit_session_map = _get_subject_visits(
            participant_id,
            subjects_visits_df,
        )
        visit_dosage_map = _get_subject_dosages(
            participant_id, subjects_visits_df, standardized_scan_dates
        )

        session_data_tuple = _combine_session_data(
            participant_id, visit_session_map, standardized_scan_dates, visit_dosage_map
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

    cross_validate_folders(bids_dir, temp_dir, subjects)

    if delete_temp_dir:
        shutil.rmtree(temp_dir)
