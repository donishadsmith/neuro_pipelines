import itertools, re
from pathlib import Path

import pandas as pd

SCORE_COLUMNS = [
    "RawScore",
    "Theta",
    "SE",
    "TScore",
    "Computed Score",
    "Uncorrected Standard Score",
    "Age-Corrected Standard Score",
    "National Percentile (age adjusted)",
    "Fully-Corrected T-score",
]


def shorten_instrument_names(unorganized_df):
    unorganized_df = unorganized_df.copy()

    instrument_names = unorganized_df["Inst"].unique().tolist()
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
    unorganized_df["Inst"] = unorganized_df["Inst"].replace(mapped_names)

    return unorganized_df, reduced_instrument_names


def run_pipeline(
    unorganized_nih_toolbox_file,
    dst_dir,
    prefix_filename,
    drop_dates,
    preexisting_nih_toolbox_file,
):
    unorganized_nih_toolbox_file = Path(unorganized_nih_toolbox_file)
    unorganized_df = pd.read_csv(
        unorganized_nih_toolbox_file, sep=None, engine="python"
    )

    unorganized_df = unorganized_df.loc[unorganized_df["PIN"] != "Test", :]

    unorganized_df = unorganized_df.sort_values(
        by=["PIN", "DateFinished"], ascending=[True, True]
    )
    unorganized_df["DateFinished"] = unorganized_df["DateFinished"].apply(
        lambda x: x if pd.isna(x) else x.split()[0]
    )
    visits_list = (
        unorganized_df[["PIN", "Assessment Name", "DateFinished"]]
        .drop_duplicates()
        .dropna()
        .values.tolist()
    )

    unorganized_df, instrument_names = shorten_instrument_names(unorganized_df)

    participant_ids, session_ids, assessment_dates = zip(*visits_list)
    data_dict = {
        "Participant ID": list(participant_ids),
        "Session ID": list(session_ids),
        "Assessment Date": assessment_dates,
    }

    products = list(itertools.product(instrument_names, SCORE_COLUMNS))
    for instrument_name, score_name in products:
        data_dict[f"{instrument_name} {score_name}"] = []

    for participant_id in sorted(set(data_dict["Participant ID"])):
        for instrument_name, score_name in products:
            values = unorganized_df.loc[
                (unorganized_df["PIN"] == participant_id)
                & (unorganized_df["Inst"] == instrument_name),
                score_name,
            ].tolist()
            if not values:
                values = [float("NaN")] * unorganized_df.loc[
                    unorganized_df["PIN"] == participant_id, "Assessment Name"
                ].nunique()

            data_dict[f"{instrument_name} {score_name}"].extend(values)

    organized_df = pd.DataFrame(data_dict)
    organized_df = organized_df.dropna(axis=1)

    if preexisting_nih_toolbox_file and Path(preexisting_nih_toolbox_file).exists():
        if preexisting_nih_toolbox_file.endswith(".xlsx"):
            preexisting_organized_df = pd.read_excel(preexisting_nih_toolbox_file)
        else:
            preexisting_organized_df = pd.read_csv(
                preexisting_nih_toolbox_file, sep=None, engine="python"
            )

        if "Assessment Date" not in preexisting_organized_df.columns:
            organized_df = organized_df.drop(columns=["Assessment Date"])

        organized_df = pd.concat(
            [preexisting_organized_df, organized_df], axis=0, ignore_index=True
        )
        organized_df = organized_df.drop_duplicates()

    if drop_dates and "Assessment Date" in organized_df.columns:
        organized_df = organized_df.drop(columns=["Assessment Date"])

    prefix_filename = f"{prefix_filename}_" if prefix_filename else ""
    output_dir = Path(dst_dir) if dst_dir else unorganized_nih_toolbox_file.parent
    output_filename = output_dir / f"{prefix_filename}organized_nih_toolbox_data.csv"
    organized_df.to_csv(output_filename, sep=",", index=False)

    return output_filename
