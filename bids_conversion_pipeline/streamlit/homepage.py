import streamlit as st

st.title("BIDS Conversion App")

st.markdown("Select a pipeline from the sidebar.")

st.markdown("**Available Pipelines:**")
st.markdown(
    "1. **BIDS Conversion** - Convert a source dataset to BIDS format.\n"
    "2. **Participants TSV** - Create or update the participants TSV file.\n"
    "3. **Add Dosages** - Add dosage information to sessions TSV files."
)