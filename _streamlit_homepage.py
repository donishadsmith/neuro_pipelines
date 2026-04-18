import streamlit as st

st.title("\U0001f9e0 Neuro Pipelines App Hub \U0001f9e0")

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
st.page_link(
    "move_files/move_files_app.py",
    label=(
        "4. **Move Files** - Move BIDS events or sessions files to a BIDS-compliant directory."
    ),
)
st.page_link(
    "events/bids_events_app.py",
    label=("5. **BIDS Events** - Create BIDS-compliant events TSV files."),
)
st.page_link(
    "connors/connors_app.py",
    label=("6. **Connors 4** - Extract Connors 4 Scores from PDF files."),
)
st.divider()
