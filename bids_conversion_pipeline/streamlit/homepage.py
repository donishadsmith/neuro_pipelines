import streamlit as st

st.title("BIDS Conversion App")

st.divider()

st.markdown("**Available Pipelines:**")
st.page_link(
    "bids_conversion_pipeline/streamlit/bids_conversion_app.py",
    label=("1. **BIDS Conversion** - Convert a source dataset to BIDS format."),
)
st.page_link(
    "bids_conversion_pipeline/streamlit/participants_tsv_app.py",
    label=("2. **Participants TSV** - Create or update the participants TSV file."),
)
st.page_link(
    "bids_conversion_pipeline/streamlit/add_dosages_app.py",
    label=("3. **Add Dosages** - Add dosage information to sessions TSV files."),
)
