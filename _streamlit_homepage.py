import streamlit as st

st.title("Neuro Pipelines App Hub")

st.markdown("Select a pipeline from the sidebar.")

st.markdown("**Available Pipelines:**")
st.markdown(
    "1. **BIDS Conversion** - Convert a source dataset to BIDS format.\n"
    "2. **Participants TSV** - Create or update the participants TSV file.\n"
    "3. **Add Dosages** - Add dosage information to sessions TSV files.\n"
    "4. **Move Files** - Move BIDS events or sessions files to a BIDS-compliant directory.\n"
    "5. **BIDS Events** - Create BIDS-compliant events TSV files.\n"
    "6. **Connors 4** - Extract Connors 4 Scores from PDF files."
)