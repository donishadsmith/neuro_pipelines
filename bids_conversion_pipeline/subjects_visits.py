import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from bidsaid.io import regex_glob
from bidsaid.path_utils import is_valid_date
from bidsaid.logging import setup_logger

from _general_utils import guess_delimiter, _standardize_dates

LGR = setup_logger(__name__)


def run_pipeline(src_dir, subjects_visits_file, output_dir=None):
    if subjects_visits_file and Path(subjects_visits_file).exists():
        if subjects_visits_file.endswith(".xlsx"):
            is_xlsx = True
            subjects_visits_df = pd.read_excel(subjects_visits_file)
        else:
            is_xlsx = False
            sep = guess_delimiter(subjects_visits_file)
            subjects_visits_df = pd.read_csv(subjects_visits_file, sep=sep)
    else:
        subjects_visits_df = pd.DataFrame({"participant_id": [], "date": []})

    if not output_dir:
        if subjects_visits_file and Path(subjects_visits_file).exists():
            output_dir = Path(subjects_visits_file).parent
        else:
            output_dir = Path().home()
    else:
        output_dir = Path(output_dir)

    if not all(x in subjects_visits_df.columns for x in ["participant_id", "date"]):
        raise ValueError(
            "Both 'participant_id' and 'date' must be columns in `subjects_visits_file`."
        )

    subjects_visits_df = _standardize_dates(subjects_visits_df)

    subject_folders = regex_glob(src_dir, pattern=r"^\d+.*_\d+$")
    subject_id_date_tuples = [
        (x.name.split("_")[0], x.name.split("_")[1]) for x in subject_folders
    ]
    data = {"participant_id": [], "date": []}
    for subject_id, scan_date in subject_id_date_tuples:
        try:
            # KKI likely has a pipeline that automatically labels folders with the "%y%m%d" date format
            new_date = (
                str(pd.to_datetime([scan_date], format=r"%y%m%d")[0]).split()[0]
                if is_valid_date(scan_date, r"%y%m%d")
                else str(pd.to_datetime([scan_date])[0]).split()[0]
            )
        except:
            LGR.warning(
                f"For subject {subject_id}, the following date is invalid: {scan_date}"
            )
            continue

        if subjects_visits_df[
            (subjects_visits_df["participant_id"] == subject_id)
            & (subjects_visits_df["date"] == new_date)
        ].empty:
            data["participant_id"].append(subject_id)
            data["date"].append(new_date)

    df = pd.DataFrame(data)
    if not df.empty:
        if not subjects_visits_df.empty:
            subjects_visits_df = pd.concat([subjects_visits_df, df], axis=0)
            subjects_visits_df = subjects_visits_df.sort_values(
                by=["participant_id", "date"], ascending=True
            )
            if is_xlsx:
                subjects_visits_df.to_excel(
                    output_dir / Path(subjects_visits_file).name
                )
            else:
                subjects_visits_df.to_csv(
                    output_dir / Path(subjects_visits_file).name, sep=sep, index=False
                )
        else:
            subjects_visits_file = output_dir / "subjects_visits_file.csv"
            LGR.info(f"New subjects visits file created at: {subjects_visits_file}")
            df.to_csv(subjects_visits_file, sep=",", index=None)
    else:
        LGR.info("No new dates appended to `subjects_visits_file`.")

    return subjects_visits_file
