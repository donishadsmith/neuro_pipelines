import logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "connors"))

import streamlit as st

from _streamlit_utils import StreamlitLogHandler, _select_content

from get_connors_score import run_pipeline

st.set_page_config(layout="centered")

st.title("Connors 4 Score Extraction Pipeline")

st.divider()

st.markdown("**Required Arguments**")

if st.button(
    "Browse for source (Connors 4 PDF) directory",
    help="Directory containing the Connors 4 PDF file.",
):
    folder = _select_content("directory")
    if folder:
        st.session_state.pdf_dir = folder
        st.session_state.has_pdf_files = bool(list(Path(folder).glob("*.pdf")))

if st.session_state.get("pdf_dir"):
    if st.session_state.has_pdf_files:
        st.success(f"Connors 4 Source Directory: {st.session_state.pdf_dir}")
    else:
        st.error(f"No PDF files detected: {st.session_state.pdf_dir}")

st.divider()
st.markdown("**Optional Arguments**")
if st.button(
    "Browse for CSV/Excel file",
    help="File path for CSV/Excel file containing Conners 4 data. If CSV/Excel file exists.",
):
    file = _select_content("file")
    if file:
        st.session_state.csv_file_path = file

if st.session_state.get("csv_file_path"):
    st.success(f"Connors 4 CSV file: {st.session_state.csv_file_path}")

include_assessment_dates = st.checkbox(
    "Include assessment dates",
    value=True,
    help="Include the date of Connors 4 was issued to participant",
)

kwargs = {
    "pdf_dir": st.session_state.get("pdf_dir"),
    "csv_file_path": st.session_state.get("csv_file_path"),
    "include_assessment_dates": include_assessment_dates,
}

st.divider()
if st.button("Run Pipeline", type="primary"):
    if not st.session_state.get("pdf_dir"):
        st.error("Please select a source (Connors 4 PDF) directory before running.")
    else:
        status_container = st.empty()
        with status_container.status("Running pipeline...", expanded=True) as status:
            handler = StreamlitLogHandler(status)
            logging.getLogger().addHandler(handler)

            output_path = run_pipeline(**kwargs)

            logging.getLogger().removeHandler(handler)

            if output_path:
                status.update(
                    label=f"Data saved to: {output_path}",
                    state="complete",
                    expanded=False,
                )
            else:
                status.update(
                    label=f"No PDF files found in: {st.session_state.pdf_dir}",
                    state="error",
                    expanded=False,
                )
