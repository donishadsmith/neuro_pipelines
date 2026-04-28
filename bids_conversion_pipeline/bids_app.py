import streamlit as st

homepage = st.Page("streamlit/homepage.py", title="Home")
bids_conversion = st.Page(
    "streamlit/bids_conversion_app.py",
    title="NIfTI to BIDS Dataset Conversion Pipeline",
)
participants_tsv = st.Page(
    "streamlit/participants_tsv_app.py", title="Participants TSV Pipeline"
)
participants_demographics = st.Page(
    "streamlit/participants_demographics_app.py",
    title="Participants Demographics Pipeline",
)
add_dosages = st.Page("streamlit/add_dosages_app.py", title="Add Dosages Pipeline")

pg = st.navigation(
    [
        homepage,
        bids_conversion,
        participants_tsv,
        participants_demographics,
        add_dosages,
    ]
)
pg.run()
