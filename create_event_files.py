import argparse, shutil, tempfile
from pathlib import Path

import pandas as pd
import numpy as np

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
from nifti2bids.metadata import parse_date_from_path


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
        required=False,
        default=None,
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
        help="The id of the subject without 'sub-'.",
    )
    parser.add_argument(
        "--delete_temp_dir",
        dest="delete_temp_dir",
        required=False,
        default=False,
        help="Deletes the temporary directory.",
    )

    return parser


def _filter_log_files(log_files, subjects):
    if subjects:
        return [
            log_file
            for log_file in log_files
            if any(subject in log_file.name for subject in subjects)
        ]
    else:
        return log_files


def _get_presentation_session(src_dir, subject_id, excel_file):
    file_dates = [
        parse_date_from_path(path, "%Y%m%d")
        for path in list(src_dir.glob(f"*{subject_id}*"))
    ]
    session_id = [date in str(excel_file.name) for date in file_dates].index(True) + 1

    return session_id


def save_df_as_tsv(event_df, dst_dir, subject_id, session_id, task):
    tsv_filename = (
        dst_dir / f"sub-{subject_id}_ses-0{session_id}_task-{task}_run-01_events.tsv"
    )
    event_df.to_csv(tsv_filename, sep="\t", index=False)


def _create_flanker_events_files(src_dir, dst_dir, subjects):
    excel_files = _filter_log_files(src_dir.glob("*.xls"), subjects)
    for excel_file in excel_files:
        extractor = PresentationEventExtractor(
            excel_file,
            convert_to_seconds=["Time", "Duration"],
            trial_types=(
                "congruentleft",
                "congruentright",
                "incongruentright",
                "incongruentleft",
                "nogoleft",
                "nogoright",
                "neutralleft",
                "neutralright",
            ),
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
                "miss": "correct",
                "incorrect": "incorrect",
                "other": "correct",
                "false_alarm": "incorrect",
                "false alarm": "incorrect",
            }
        )
        event_df = pd.DataFrame(events)

        # Specific accuracy case for nogo
        event_df.loc[
            (event_df["trial_type"] == "nogo") & (event_df["response"] == "miss"),
            "accuracy",
        ] = "correct"

        event_df["trial_type_accuracy"] = (
            event_df["trial_type"].astype(str) + "_" + event_df["accuracy"].astype(str)
        )

        # Getting subject ID and organising files to get subject ID
        subject_id = str(excel_file.name).split("_")[0]
        session_id = _get_presentation_session(src_dir, subject_id, excel_file)

        save_df_as_tsv(event_df, dst_dir, subject_id, session_id, task="flanker")


def _create_nback_events_files(src_dir, dst_dir, subjects):
    edat_files = _filter_log_files(src_dir.glob("*.edat3"), subjects)
    for edat_file in edat_files:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmpfile:
            pass

        try:
            csv_path = convert_edat3_to_text(edat_file, dst_path=tmpfile.name)

            input_df = pd.read_csv(csv_path, sep=",")
            input_df["Procedure[Block]"] = input_df["Procedure[Block]"].map(
                {"ExpBloc": "1-back", "ContBloc": "0-back", "Exp2Bloc": "2-back"}
            )

            extractor = EPrimeBlockExtractor(
                input_df,
                onset_column_name="StimDisplay.OnsetTime",
                procedure_column_name="Procedure[Block]",
                block_cue_names=("1-back", "0-back", "2-back"),
                convert_to_seconds=["StimDisplay.OnsetTime", "StimDisplay.RT"],
                rest_block_code="Rest",
                rest_code_frequency="variable",
            )
            events = {}
            # No onset column timing infomation, make assumption that as soon as scanner started
            # immediately starts at the first rest block which occures 16 seconds prior to the
            # first experimental block
            first_stim_onset_time = (
                input_df["StimDisplay.OnsetTime"].dropna(inplace=False).values[0] / 1e3
            )
            scanner_onset_time = first_stim_onset_time - 16

            events["onset"] = extractor.extract_onsets(
                scanner_start_time=scanner_onset_time
            )
            events["duration"] = extractor.extract_durations()
            events["trial_type"] = extractor.extract_trial_types()
            event_df = pd.DataFrame(events)
            # No quit code and time of rest block is never recorded so clip to 37.0 duration
            event_df["duration"] = event_df["duration"].apply(
                lambda x: x if x < 38.0 and x > 35.0 and not np.isnan(x) else 37.0
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

            subject_id, session_id = (
                str(edat_file.name).removesuffix(".edat3").split("-")[1:]
            )

            save_df_as_tsv(event_df, dst_dir, subject_id, session_id, task="nback")
        finally:
            csv_path.unlink()


def _create_mtl_events_files(src_dir, dst_dir, subjects, task):
    # MTLE and MTLR are separate tasks but can be processed in one function
    filename = "_PEARencN" if task == "mtle" else "_PEARretN"
    task_name = "indoor" if task == "mtle" else "seen"

    excel_files = _filter_log_files(src_dir.glob(f"*{filename}*.xls"), subjects)
    for excel_file in excel_files:
        input_df = load_presentation_log(excel_file)
        # Add quit code, some log files did not record the quit event type
        if not input_df[input_df["Event Type"] == "Quit"].empty:
            input_df.loc[input_df["Event Type"] == "Quit", "Code"] = "quit"

        extractor = PresentationBlockExtractor(
            input_df,
            convert_to_seconds=["Time"],
            block_cue_names=(task_name),
            scanner_event_type="Pulse",
            scanner_trigger_code="30",
            rest_block_code="rest",
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

        subject_id = str(excel_file.name).split("_")[0]
        session_id = _get_presentation_session(src_dir, subject_id, excel_file)

        save_df_as_tsv(event_df, dst_dir, subject_id, session_id, task)


def _create_princess_events_files(src_dir, dst_dir, subjects):
    edat_files = _filter_log_files(src_dir.glob("*.edat3"), subjects)
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

            huizens = []
            for huizen in input_df["huizen"].astype(str).values:
                huizen = huizen.removesuffix(".bmp")
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
                convert_to_seconds=["dagnacht.OnsetTime", "eind.OnsetTime"],
                split_cue_as_instruction=True,
            )

            events = {}
            # Best guess of scanner time would be the first fixpunt which appears after
            # scanner sends trigger
            scanner_start_time = (
                input_df["fixpunt.OnsetTime"].dropna().unique()[0] / 1e3
            )
            events["onset"] = extractor.extract_onsets(
                scanner_start_time=scanner_start_time
            )
            events["duration"] = extractor.extract_durations()
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
            event_df.loc[event_df.index[-1], "duration"] = (
                event_df.loc[event_df.index[-1], "onset"]
                - input_df["eind.OnsetTime"].values[0] / 1e3
            )

            subject_id, session_id = (
                str(edat_file.name).removesuffix(".edat3").split("-")[1:]
            )
            save_df_as_tsv(event_df, dst_dir, subject_id, session_id, task="princess")
        finally:
            csv_path.unlink()


def main(src_dir, dst_dir, temp_dir, task, subjects, delete_temp_dir):
    func = {
        "flanker": _create_flanker_events_files,
        "nback": _create_nback_events_files,
        "mtle": _create_mtl_events_files,
        "mtlr": _create_mtl_events_files,
        "princess": _create_princess_events_files,
    }

    if task not in func:
        raise ValueError(f"`task` must be one of the following: {func.keys()}")

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not dst_dir.exists():
        dst_dir.mkdir()

    if temp_dir:
        temp_dir = Path(temp_dir)

    kwargs = {
        "src_dir": (temp_dir or src_dir),
        "dst_dir": dst_dir,
        "subjects": subjects,
    }
    if task in ["mtle", "mtlr"]:
        kwargs.update({"task": task})

    try:
        if temp_dir:
            temp_dir = Path(temp_dir)
            shutil.copytree(src_dir, temp_dir)
            func[task](**kwargs)
    finally:
        if temp_dir and temp_dir.exists() and delete_temp_dir:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    cmd_args = _get_cmd_args()
    args = cmd_args.parse_args()
    main(**vars(args))
