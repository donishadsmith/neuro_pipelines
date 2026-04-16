import streamlit as st

st.title("BIDS Conversion App")

st.markdown("Select a pipeline from the sidebar.")

st.markdown("**Available Pipelines:**")
st.markdown(
    "1. **BIDS Conversion** — Convert a source dataset to BIDS format.\n"
    "2. **Participants TSV** — Create or update the participants TSV file.\n"
    "3. **Add Dosages** — Add dosage information to sessions TSV files."
)

bids_conversion = st.Page(
    "streamlit/bids_conversion_app.py", title="BIDS Conversion Pipeline"
)
participants_tsv = st.Page(
    "streamlit/participants_tsv_app.py", title="Participants TSV Pipeline"
)
add_dosages = st.Page("streamlit/add_dosages_app.py", title="Add Dosages Pipeline")

pg = st.navigation([bids_conversion, participants_tsv, add_dosages])
pg.run()
