import argparse, os, shutil, tempfile
from pathlib import Path
from datetime import datetime

import numpy as np, pandas as pd

from nifti2bids.parsers import (
    load_presentation_log,
    convert_edat3_to_text,
)
from nifti2bids.bids import (
    PresentationBlockExtractor,
    PresentationEventExtractor,
    EPrimeBlockExtractor,
    add_instruction_timing,
)
from nifti2bids.io import _copy_file
from nifti2bids.logging import setup_logger
from nifti2bids.metadata import is_valid_date

LGR = setup_logger(__name__)


def _get_cmd_args():
    parser = argparse.ArgumentParser(description="Create BIDs compliant events files.")
    parser.add_argument(
        "--src_dir",
        dest="src_dir",
        required=True,
        help="Path to directory containing neurobehavioral log data.",
    )
    parser.add_argument(
        "--dst_dir",
        dest="dst_dir",
        required=True,
        help="Path to destination directory to output event files to.",
    )
    parser.add_argument(
        "--temp_dir",
        dest="temp_dir",
        required=True,
        help="Path to a temporary directory to use.",
    )
    parser.add_argument(
        "--task",
        dest="task",
        required=True,
        help="The name of the task (i.e., 'nback', 'flanker', 'mtle', 'mtlr', 'princess')",
    )
    parser.add_argument(
        "--subjects",
        dest="subjects",
        required=False,
        default=None,
        nargs="+",
        help="The ID of the subject without 'sub-'.",
    )
    parser.add_argument(
        "--minimum_file_size",
        dest="minimum_file_size",
        required=False,
        default=None,
        help="The minimum file size in bytes to ignore error files.",
    )
    # Extracting the file creation or modification date may not be very reliable
    parser.add_argument(
        "--subjects_visits_file",
        dest="subjects_visits_file",
        required=False,
        default=None,
        help=(
            "A text file, where the 'subject_id' contains the subject ID and the "
            "'date' column is the date of visit. Using this parameter is recommended "
            "when data is missing. Ensure all dates have a consistent format. "
            "**All subject visit dates should be listed.**"
        ),
    )
    parser.add_argument(
        "--subjects_visits_date_fmt",
        dest="subjects_visits_date_fmt",
        required=False,
        default=r"%m/%d/%Y",
        help=("The format of the date in the ``subjects_visits`` file."),
    )
    parser.add_argument(
        "--src_data_date_fmt",
        dest="src_data_date_fmt",
        required=False,
        default=r"%Y%m%d",
        help=(
            "The format of the dates in the filenames that are in the source directory."
        ),
    )

    return parser


class SubjectsVisitsFileError(Exception):
    """Exception for issues with the subjects sessions file."""

    pass


def _filter_log_files(log_files, subjects):
    if subjects:
        return [
            log_file
            for log_file in log_files
            if any(subject in log_file.name for subject in subjects)
        ]
    else:
        return log_files


def _get_minimum_file_size(task, minimum_file_size):
    file_size_minimum_kb = {
        "flanker": 40,
        "mtle": 5,
        "mtlr": 5,
        "princess": 50,
        "nback": 40,
    }

    if not minimum_file_size:
        minimum_file_size = file_size_minimum_kb[task] * 1024

    return minimum_file_size


def _copy_event_files(src_dir, temp_dir, task, minimum_file_size):
    for event_file in src_dir.glob("*"):
        minimum_file_size = _get_minimum_file_size(task, minimum_file_size)

        if os.path.getsize(event_file) < minimum_file_size:
            LGR.critical(
                f"The following file is smaller than {minimum_file_size} bytes "
                f"and will not be copied to the temporary directory: {event_file}."
            )

            continue

        _copy_file(event_file, temp_dir / event_file.name, remove_src_file=False)


def _check_subjects_visits_file(subjects_visits_file: str | Path) -> None:
    required_colnames = ["subject_id", "date"]
    colnames = pd.read_csv(subjects_visits_file, sep=None, engine="python").columns
    if not all(required_colname in colnames for required_colname in required_colnames):
        raise SubjectsVisitsFileError(
            f"The following columns are required in {subjects_visits_file}: "
            f"{required_colnames}."
        )


def _get_subjects_visits(
    subject_id, subjects_visits_df, subjects_visits_date_fmt, src_data_date_fmt
):
    # Don't sort to keep the order of the NaNs
    visit_dates = (
        subjects_visits_df[subjects_visits_df["subject_id"].astype(str) == subject_id]
        .loc[:, "date"]
        .to_numpy(copy=True)
        .tolist()
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
        date: session_id
        for session_id, date in enumerate(list(map(convert_date, visit_dates)), start=1)
    }


def _get_presentation_session(
    temp_dir,
    subject_id,
    excel_file,
    task=None,
    subjects_visits_df=None,
    subjects_visits_date_fmt=None,
    src_data_date_fmt=None,
):
    if task in ["mtle", "mtlr"]:
        identifier = "_PEARencN" if task == "mtle" else "_PEARretN"
        file_dates = sorted(
            [
                path.name.split("_")[-2]
                for path in list(temp_dir.glob(f"*{subject_id}*{identifier}*"))
            ]
        )
    else:
        file_dates = sorted(
            [
                path.name.split("_")[-2]
                for path in list(temp_dir.glob(f"*{subject_id}*"))
            ]
        )

    if not all(is_valid_date(date, src_data_date_fmt) for date in file_dates):
        LGR.warning(
            f"Not all dates have the following format ({src_data_date_fmt}) "
            f"for subject {subject_id}: {file_dates}."
        )

    visit_session_map = (
        _get_subjects_visits(
            subject_id, subjects_visits_df, subjects_visits_date_fmt, src_data_date_fmt
        )
        if subjects_visits_df is not None
        else None
    )
    if visit_session_map:
        curr_date = [date for date in file_dates if date in excel_file.name][0]
        date_in_map = curr_date in visit_session_map
        if not date_in_map:
            LGR.critical(
                f"Subject {subject_id} does not have the following date: {curr_date}. "
                "The date will be used as the session label."
            )

        return visit_session_map[curr_date] if date_in_map else curr_date
    else:
        return [date in excel_file.name for date in file_dates].index(True) + 1


def save_df_as_tsv(event_df, dst_dir, subject_id, session_id, task):
    tsv_filename = (
        dst_dir / f"sub-{subject_id}_ses-0{session_id}_task-{task}_run-01_events.tsv"
    )
    event_df.to_csv(tsv_filename, sep="\t", index=False)


def _create_flanker_events_files(
    temp_dir,
    dst_dir,
    subjects,
    subjects_visits_df,
    subjects_visits_date_fmt,
    src_data_date_fmt,
):
    excel_files = _filter_log_files(temp_dir.glob("*.xls"), subjects)
    for excel_file in excel_files:
        extractor = PresentationEventExtractor(
            excel_file,
            convert_to_seconds=["Time", "Duration"],
            trial_types=".*(left|right)$",
            scanner_event_type="Pulse",
            scanner_trigger_code="99",
        )

        events = {}
        events["onset"] = extractor.extract_onsets()
        events["duration"] = extractor.extract_durations()

        # Separate trial name from arrow direction
        info = []
        for trial_type in extractor.extract_trial_types():
            trial_name = trial_type.removesuffix("left").removesuffix("right")
            arrow_dir = trial_type.removeprefix(trial_name)
            info.append((trial_name, arrow_dir))

        trial_types, arrow_dirs = zip(*info)

        events["trial_type"] = trial_types
        events["central_arrow_direction"] = arrow_dirs
        events["response"] = extractor.extract_responses()
        events["accuracy"] = extractor.extract_accuracies(
            {
                "hit": "correct",
                "miss": "incorrect",
                "incorrect": "incorrect",
                "other": "correct",
                "false_alarm": "incorrect",
                "false alarm": "incorrect",
            }
        )
        event_df = pd.DataFrame(events)

        # Specific accuracy case for miss
        event_df.loc[
            (event_df["trial_type"] == "nogo") & (event_df["response"] == "miss"),
            "accuracy",
        ] = "correct"

        event_df["trial_type_accuracy"] = (
            event_df["trial_type"].astype(str) + "_" + event_df["accuracy"].astype(str)
        )

        # Getting subject ID and organising files to get subject ID
        subject_id = excel_file.name.split("_")[0]
        session_id = _get_presentation_session(
            temp_dir,
            subject_id,
            excel_file,
            subjects_visits_df=subjects_visits_df,
            subjects_visits_date_fmt=subjects_visits_date_fmt,
            src_data_date_fmt=src_data_date_fmt,
        )

        save_df_as_tsv(event_df, dst_dir, subject_id, session_id, task="flanker")


def _create_nback_events_files(temp_dir, dst_dir, subjects):
    edat_files = _filter_log_files(temp_dir.glob("*.edat3"), subjects)
    for edat_file in edat_files:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmpfile:
            pass

        try:
            csv_path = convert_edat3_to_text(edat_file, dst_path=tmpfile.name)

            input_df = pd.read_csv(csv_path, sep=",")
            input_df["StimDisplay.OffsetTime"] = input_df[
                "StimDisplay.OnsetTime"
            ].to_numpy(copy=True) + input_df["StimDisplay.OnsetToOnsetTime"].to_numpy(
                copy=True
            )
            input_df["Procedure[Block]"] = input_df["Procedure[Block]"].fillna("Rest")
            input_df["Procedure[Block]"] = input_df["Procedure[Block]"].map(
                {
                    "ExpBloc": "1-back",
                    "ContBloc": "0-back",
                    "Exp2Bloc": "2-back",
                    "Rest": "Rest",
                }
            )
            input_df.loc[input_df.index[-1], "Procedure[Block]"] = "Quit"

            extractor = EPrimeBlockExtractor(
                input_df,
                onset_column_name="StimDisplay.OnsetTime",
                procedure_column_name="Procedure[Block]",
                block_cue_names=("1-back", "0-back", "2-back"),
                convert_to_seconds=["StimDisplay.OnsetTime", "StimDisplay.OffsetTime"],
                rest_block_codes="Rest",
                quit_code="Quit",
                rest_code_frequency="variable",
            )
            events = {}
            # No onset column timing infomation, make assumption that as soon as scanner started
            # immediately starts at the first rest block which occures 16 seconds prior to the
            # first experimental block
            first_stim_onset_time = (
                input_df["StimDisplay.OnsetTime"]
                .dropna(inplace=False)
                .to_numpy(copy=True)[0]
                / 1e3
            )
            scanner_onset_time = first_stim_onset_time - 16

            events["onset"] = extractor.extract_onsets(
                scanner_start_time=scanner_onset_time
            )
            events["duration"] = extractor.extract_durations(
                offset_column_name="StimDisplay.OffsetTime"
            )
            events["trial_type"] = extractor.extract_trial_types()
            event_df = pd.DataFrame(events)

            event_df["duration"] = event_df["duration"].apply(
                lambda x: x if not np.isnan(x) else 34.0
            )

            # Split instruction block, which is 2 seconds before each stimulus
            event_df = add_instruction_timing(event_df, instruction_duration=2)
            event_df["trial_type"] = event_df["trial_type"].replace(
                {
                    "1-back_instruction": "instruction",
                    "2-back_instruction": "instruction",
                    "0-back_instruction": "instruction",
                }
            )

            subject_id, session_id = edat_file.name.removesuffix(".edat3").split("-")[
                1:
            ]

            save_df_as_tsv(event_df, dst_dir, subject_id, session_id, task="nback")
        finally:
            csv_path.unlink()


def _create_mtl_events_files(
    temp_dir,
    dst_dir,
    subjects,
    task,
    subjects_visits_df,
    subjects_visits_date_fmt,
    src_data_date_fmt,
):
    # MTLE and MTLR are separate tasks but can be processed in one function
    filename = "_PEARencN" if task == "mtle" else "_PEARretN"
    task_name = "indoor" if task == "mtle" else "seen"

    excel_files = _filter_log_files(temp_dir.glob(f"*{filename}*.xls"), subjects)
    for excel_file in excel_files:
        input_df = load_presentation_log(excel_file)
        # Add quit code, some log files did not record the quit event type
        if not input_df[input_df["Event Type"] == "Quit"].empty:
            input_df.loc[input_df["Event Type"] == "Quit", "Code"] = "quit"

        extractor = PresentationBlockExtractor(
            input_df,
            convert_to_seconds=["Time", "Duration"],
            block_cue_names=(task_name),
            scanner_event_type="Pulse",
            scanner_trigger_code="30",
            rest_block_codes="rest",
            quit_code="quit",
            split_cue_as_instruction=True,
        )

        events = {}
        events["onset"] = extractor.extract_onsets()
        durations = extractor.extract_durations()
        # Fix for those that do not have the quit event type
        durations[-1] = durations[-1] if durations[-1] != 0 else 20.0
        events["duration"] = durations
        events["trial_type"] = extractor.extract_trial_types()

        event_df = pd.DataFrame(events)
        event_df["trial_type"] = event_df["trial_type"].replace(
            {f"{task_name}_instruction": "instruction"}
        )

        # Special case for subject 10308 to get subject and session
        subject_id = excel_file.name.split("_")[0]
        session_id = _get_presentation_session(
            temp_dir,
            subject_id,
            excel_file,
            task,
            subjects_visits_df,
            subjects_visits_date_fmt,
            src_data_date_fmt,
        )

        save_df_as_tsv(event_df, dst_dir, subject_id, session_id, task)


def _create_princess_events_files(temp_dir, dst_dir, subjects):
    edat_files = _filter_log_files(temp_dir.glob("*.edat3"), subjects)
    for edat_file in edat_files:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmpfile:
            pass

        try:
            csv_path = convert_edat3_to_text(edat_file, dst_path=tmpfile.name)

            # Moving cue time to onset column
            input_df = pd.read_csv(csv_path, sep=",")
            input_df.loc[
                ~input_df["indicatie.OnsetTime"].isna(), "dagnacht.OnsetTime"
            ] = input_df.loc[
                ~input_df["indicatie.OnsetTime"].isna(), "indicatie.OnsetTime"
            ]

            # Based on original paper, trials blocks should be ~52 seconds each
            # Still derive to check if timing is padded
            # Note, offset times were not recorded but paper states trials include the feedback
            input_df["feedback.OffsetTime"] = input_df["feedback.OnsetTime"].to_numpy(
                copy=True
            ) + input_df["feedback.OnsetToOnsetTime"].to_numpy(copy=True)

            huizens = []
            for huizen in input_df["huizen"].astype(str).to_numpy(copy=True):
                huizen = str(huizen).removesuffix(".bmp")
                if huizen[-1].isdigit():
                    huizen = huizen[:-1]
                huizens.append(huizen)

            input_df["huizen"] = huizens

            dutch_to_english = {"dag": "day", "nacht": "night", "dagnacht": "daynight"}
            input_df["huizen"] = input_df["huizen"].replace(dutch_to_english)
            extractor = EPrimeBlockExtractor(
                input_df,
                onset_column_name="dagnacht.OnsetTime",
                procedure_column_name="huizen",
                trigger_column_name="eind.OnsetTime",
                block_cue_names=dutch_to_english.values(),
                convert_to_seconds=[
                    "dagnacht.OnsetTime",
                    "eind.OnsetTime",
                    "feedback.OffsetTime",
                ],
                split_cue_as_instruction=False,
            )

            events = {}
            # Best guess of scanner time would be the first fixpunt which appears after
            # scanner sends trigger, two image displays appear but only second image time recorded
            # first fixpunt has a duration of ~60 seconds, the second appears three seconds
            # before cue
            scanner_start_time = (
                input_df["fixpunt.OnsetTime"].dropna().unique()[0] / 1e3 - 60
            )
            events["onset"] = extractor.extract_onsets(
                scanner_start_time=scanner_start_time
            )
            events["duration"] = extractor.extract_durations(
                offset_column_name="feedback.OffsetTime"
            )
            trial_name_dict = {
                "daynight": "switch",
                "day": "nonswitch",
                "night": "nonswitch",
            }
            events["trial_type"] = [
                trial_name_dict[trial_type] if trial_type in trial_name_dict else "cue"
                for trial_type in extractor.extract_trial_types()
            ]
            events["block_cue"] = extractor.extract_trial_types()

            event_df = pd.DataFrame(events)

            subject_id, session_id = edat_file.name.removesuffix(".edat3").split("-")[
                1:
            ]

            # Note, duration of all trials in block without cue is 48-50 seconds and duration of cue is ~3
            # seconds with above implementation: Paper says each block has 8 trials which lasts 6500 ms:
            # Paper: https://pubmed.ncbi.nlm.nih.gov/20604616/
            save_df_as_tsv(event_df, dst_dir, subject_id, session_id, task="princess")
        finally:
            csv_path.unlink()


def _get_dataframe(subjects_visits_file):
    if not subjects_visits_file:
        return None

    return pd.read_csv(subjects_visits_file, sep=None, engine="python")


def _strip_entity(subjects):
    return [str(subject).removeprefix("sub-") for subject in subjects]


def main(
    src_dir,
    dst_dir,
    temp_dir,
    task,
    subjects,
    minimum_file_size,
    subjects_visits_file,
    subjects_visits_date_fmt,
    src_data_date_fmt,
):
    func = {
        "flanker": _create_flanker_events_files,
        "nback": _create_nback_events_files,
        "mtle": _create_mtl_events_files,
        "mtlr": _create_mtl_events_files,
        "princess": _create_princess_events_files,
    }

    if task not in func:
        raise ValueError(f"`task` must be one of the following: {func.keys()}")

    if subjects_visits_file:
        _check_subjects_visits_file(subjects_visits_file)

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not dst_dir.exists():
        dst_dir.mkdir()

    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        temp_dir.mkdir()

    if subjects:
        subjects = _strip_entity(subjects)

    kwargs = {"temp_dir": temp_dir, "dst_dir": dst_dir, "subjects": subjects}
    if task in ["mtle", "mtlr"]:
        kwargs.update({"task": task})

    # Only Presentation files contain date in filename
    if task in ["mtle", "mtlr", "flanker"]:
        kwargs.update(
            {
                "subjects_visits_df": _get_dataframe(subjects_visits_file),
                "subjects_visits_date_fmt": subjects_visits_date_fmt,
                "src_data_date_fmt": src_data_date_fmt,
            }
        )

    try:
        _copy_event_files(src_dir, temp_dir, task, minimum_file_size)
        func[task](**kwargs)
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
