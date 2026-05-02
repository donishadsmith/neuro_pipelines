import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "bids_conversion_pipeline")
)

import pandas as pd, streamlit as st

from participants_demographics import run_pipeline
from _streamlit_utils import _select_content


def _get_df(filename):
    if filename.endswith(".xlsx"):
        df = pd.read_excel(filename)
    else:
        df = pd.read_csv(filename, sep=None, engine="python")

    return df


st.set_page_config(layout="centered")

st.title("Participants Demographics Pipeline")
st.divider()

st.markdown("""**Note:**\n
- The demographics data file must have a column named "participant_id".
""")

st.divider()
st.markdown("**Required Arguments**")
if st.button(
    "Browse for participants TSV file",
    help="A BIDS compliant participants TSV ('participants.tsv') file.",
):
    file = _select_content("file")
    if file:
        st.session_state.participants_tsv_path = file
        if (
            Path(file).name == "participants.tsv"
            and "participant_id" in pd.read_csv(file, sep="\t").columns
        ):
            st.session_state.valid_participants_tsv = True
        else:
            st.session_state.valid_participants_tsv = False

if st.session_state.get("participants_tsv_path"):
    if st.session_state.valid_participants_tsv:
        st.success(f"Participants TSV file: {st.session_state.participants_tsv_path}")
    else:
        st.error(
            f"Invalid participants TSV file: {st.session_state.participants_tsv_path}"
        )

if st.button(
    "Browse for demographics file",
    help="A CSV or Excel file containing subject demographics. Must have a 'participant_id' column.",
):
    file = _select_content("file")
    if file:
        st.session_state.demographics_file = file
        if "participant_id" in _get_df(file).columns:
            st.session_state.valid_demographics_file = True
        else:
            st.session_state.valid_demographics_file = False


if st.session_state.get("demographics_file"):
    if st.session_state.valid_demographics_file:
        st.success(f"Demographics file: {st.session_state.demographics_file}")
    else:
        st.error(f"Invalid demographics file: {st.session_state.demographics_file}")

if filename := st.session_state.get("demographics_file"):
    df = _get_df(filename)

    columns = [
        col
        for col in df.columns
        if col not in ["participant_id", "date", "dose", "dose_mg", "PID", "ID"]
    ]

    covariates_to_add = st.multiselect(
        "Covariates to add",
        columns,
        help="Column names from the demographics file to include in the participants TSV. Separated by commas or spaces.",
    )
else:
    covariates_to_add = None

kwargs = {
    "participants_tsv_path": st.session_state.get("participants_tsv_path"),
    "demographics_file": st.session_state.get("demographics_file"),
    "covariates_to_add": covariates_to_add,
}

st.divider()
if st.button("Run Pipeline", type="primary"):
    if not (
        st.session_state.get("participants_tsv_path")
        and st.session_state.get("valid_participants_tsv")
    ):
        st.error("Please select a valid participants TSV file before running.")
    elif not (
        st.session_state.get("demographics_file")
        and st.session_state.get("valid_demographics_file")
    ):
        st.error("Please select a valid demographics file before running.")
    elif not covariates_to_add:
        st.error(
            "Please select add the names of the demographic covariates before running."
        )
    else:
        with st.spinner("Processing..."):
            run_pipeline(**kwargs)

        st.success(
            f"Demographic data added/updated to: {st.session_state.get('participants_tsv_path')}."
        )
