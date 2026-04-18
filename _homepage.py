import streamlit as st

# Attempt to make the homepage look better with html
# Note: right click and select Inspect to see html structure
st.markdown("""
    <style>
    a[data-testid="stPageLink-NavLink"] {
        border-left: 5px solid #0097A7;
        padding-left: 10px;
    }
    a[data-testid="stPageLink-NavLink"] p {
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("\U0001f9e0 Neuro Pipelines")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.page_link("bids_conversion_pipeline/streamlit/bids_conversion_app.py", label="**NIfTI to BIDS Conversion**")
    st.caption("Convert a raw NIfTI dataset to BIDS format.")

    st.page_link("bids_conversion_pipeline/streamlit/participants_tsv_app.py", label="**Participants TSV**")
    st.caption("Create or update the participants TSV file.")

    st.page_link("bids_conversion_pipeline/streamlit/add_dosages_app.py", label="**Add Dosages**")
    st.caption("Add dosage information to sessions TSV files.")

with col2:
    st.page_link("events/bids_events_app.py", label="**BIDS Events**")
    st.caption("Create BIDS-compliant events TSV files.")

    st.page_link("move_files/move_files_app.py", label="**Move BIDS Files**")
    st.caption("Move BIDS files to a BIDS-compliant directory.")

    st.page_link("connors/connors_app.py", label="**Connors 4**")
    st.caption("Extract Connors 4 scores from PDF files.")