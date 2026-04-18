import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "connors"))

import streamlit as st

from _streamlit_utils import _select_content

from get_connors_score import run_pipeline

st.title("Connors 4 Score Extraction Pipeline")

st.divider()
st.markdown("**Required Arguments**")

if st.button(
    "Browse for source directory",
    help="Directory containing the Connors 4 PDF file.",
):
    folder = _select_content("directory")
    if folder:
        st.session_state.pdf_dir = folder

if st.session_state.get("pdf_dir"):
    st.success(f"Source: {st.session_state.pdf_dir}")

st.divider()
st.markdown("**Optional Arguments**")
if st.button(
    "Browse for CSV file",
    help="File path for CSV file containing Conners 4 data. If CSV file exists.",
):
    file = _select_content("file")
    if file:
        st.session_state.csv_file_path = file

if st.session_state.get("csv_file_path"):
    st.success(f"Source: {st.session_state.csv_file_path}")

subjects = st.text_input(
    "Subject IDs",
    help="Restrict processing to specific subjects. Enter IDs without the 'sub-' prefix, separated by commas or spaces.",
)
if subjects:
    subjects = [s.strip() for s in subjects.replace(",", " ").split() if s.strip()]

kwargs = {
    "pdf_dir": st.session_state.get("pdf_dir"),
    "csv_file_path": st.session_state.get("csv_file_path"),
    "subjects": subjects,
}

st.divider()
if st.button("Run Pipeline", type="primary"):
    if not st.session_state.get("pdf_dir"):
        st.error("Please select a source directory before running.")
    else:
        with st.spinner("Processing..."):
            dst_dir = run_pipeline(**kwargs)

        output_path = st.session_state.get("csv_file_path") or st.session_state.get(
            "pdf_dir"
        )
        st.success(f"Data saved to: {output_path}")
