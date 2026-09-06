import itertools, re, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from _general_utils import _get_dataframe

# Very lazy modifications, but making minimal changes to ensure the core logic stays intact
COLUMN_NAMES = {
    "kids": {
        "ID": "SUB",
        "Date": "Date",
        "Instruments": "InstrumentTitle",
        "Assessment": "AssessmentName",
    },
    "adults": {
        "ID": "PIN",
        "Instruments": "Inst",
        "Date": "DateFinished",
        "Assessment": "Assessment Name",
    },
}

SCORE_COLUMNS = {
    "kids": [
        "RawScore",
        "TScore",
        "TScoreStandardError",
        "Theta",
        "ThetaStandardError",
        "ChangeSensitiveScore",
        "ChangeSensitiveScoreStandardError",
        "AgeAdjustedStandardScore",
        "AgeAdjustedStandardScoreStandardError",
        "AgeEduAdjustedTScore",
        "AgeEduAdjustedTScoreStandardError",
        "FullyAdjustedTScore",
        "NationalPercentileAgeAdjusted",
        "ComputedScore",
        "ItemCount",
    ],
    "adults": [
        "RawScore",
        "Theta",
        "SE",
        "TScore",
        "Computed Score",
        "Uncorrected Standard Score",
        "Age-Corrected Standard Score",
        "National Percentile (age adjusted)",
        "Fully-Corrected T-score",
    ],
}


def clean_instrument_names(cohort, unorganized_df):
    unorganized_df = unorganized_df.copy()

    instrument_column = COLUMN_NAMES[cohort]["Instruments"]
    instrument_names = unorganized_df[instrument_column].unique().tolist()
    if cohort == "kids":
        return unorganized_df, instrument_names

    reduced_instrument_names = [
        (
            name.removeprefix("NIH Toolbox").split("Age")[0].strip()
            if name.startswith("NIH Toolbox")
            else name
        )
        for name in instrument_names
    ]
    reduced_instrument_names = [
        (
            re.split(r"v\d+\.\d+", name)[0].strip()
            if name.startswith("Cognition")
            else name
        )
        for name in reduced_instrument_names
    ]
    mapped_names = {
        k: v for k, v in list(zip(instrument_names, reduced_instrument_names))
    }
    unorganized_df[instrument_column] = unorganized_df[instrument_column].replace(
        mapped_names
    )

    return unorganized_df, reduced_instrument_names


def run_pipeline(
    cohort,
    unorganized_nih_toolbox_file,
    dst_dir,
    prefix_filename,
    include_assessment_dates,
    preexisting_nih_toolbox_file,
):
    unorganized_nih_toolbox_file = Path(unorganized_nih_toolbox_file)
    unorganized_df = pd.read_csv(
        unorganized_nih_toolbox_file, sep=None, engine="python"
    )

    instrument_column = COLUMN_NAMES[cohort]["Instruments"]
    id_column = COLUMN_NAMES[cohort]["ID"]
    date_column = COLUMN_NAMES[cohort]["Date"]
    assessment_column = COLUMN_NAMES[cohort]["Assessment"]

    unorganized_df = unorganized_df.loc[unorganized_df[id_column] != "Test", :]

    unorganized_df = unorganized_df.sort_values(
        by=[id_column, date_column], ascending=[True, True]
    )
    unorganized_df[date_column] = unorganized_df[date_column].apply(
        lambda x: x if pd.isna(x) else x.split()[0].split("T")[0]
    )
    visits_list = (
        unorganized_df[[id_column, assessment_column, date_column]]
        .drop_duplicates()
        .dropna()
        .values.tolist()
    )

    unorganized_df, instrument_names = clean_instrument_names(cohort, unorganized_df)

    participant_ids, session_ids, assessment_dates = zip(*visits_list)
    data_dict = {
        "Participant ID": list(participant_ids),
        "Session ID": list(session_ids),
        assessment_column: assessment_dates,
    }

    products = list(itertools.product(instrument_names, SCORE_COLUMNS[cohort]))
    for instrument_name, score_name in products:
        data_dict[f"{instrument_name} {score_name}"] = []

    for participant_id, session_id in zip(
        data_dict["Participant ID"], data_dict["Session ID"]
    ):
        for instrument_name, score_name in products:
            value = unorganized_df.loc[
                (unorganized_df[id_column] == participant_id)
                & (unorganized_df[assessment_column] == session_id)
                & (unorganized_df[instrument_column] == instrument_name),
                score_name,
            ].tolist()

            if not value:
                value = [float("NaN")]

            data_dict[f"{instrument_name} {score_name}"].extend(value)

    organized_df = pd.DataFrame(data_dict)

    if preexisting_nih_toolbox_file and Path(preexisting_nih_toolbox_file).exists():
        preexisting_organized_df = _get_dataframe(preexisting_nih_toolbox_file)

        if assessment_column not in preexisting_organized_df.columns:
            organized_df = organized_df.drop(columns=[assessment_column])

        organized_df = pd.concat(
            [preexisting_organized_df, organized_df], axis=0, ignore_index=True
        )
        organized_df = organized_df.drop_duplicates()

    if not include_assessment_dates and assessment_column in organized_df.columns:
        organized_df = organized_df.drop(columns=[assessment_column])

    prefix_filename = f"{prefix_filename}_" if prefix_filename else ""
    output_dir = Path(dst_dir) if dst_dir else unorganized_nih_toolbox_file.parent
    output_filename = output_dir / f"{prefix_filename}organized_nih_toolbox_data.csv"
    organized_df.to_csv(output_filename, sep=",", index=False)

    return output_filename
