import streamlit as st

homepage = st.Page("_homepage.py", title="Home")
bids_conversion = st.Page(
    "bids_conversion_pipeline/streamlit/bids_conversion_app.py",
    title="NIfTI to BIDS Conversion Pipeline",
)
participants_tsv = st.Page(
    "bids_conversion_pipeline/streamlit/participants_tsv_app.py",
    title="Participants TSV Pipeline",
)
participants_demographics = st.Page(
    "bids_conversion_pipeline/streamlit/participants_demographics_app.py",
    title="Participants Demographics Pipeline",
)
add_dosages = st.Page(
    "bids_conversion_pipeline/streamlit/add_dosages_app.py",
    title="Add Dosages Pipeline",
)
move_files = st.Page("move_files/move_files_app.py", title="Move BIDS Files Pipeline")
events = st.Page("events/bids_events_app.py", title="BIDS Events Pipeline")
behavioral_data = st.Page(
    "events/behavioral_data_app.py", title="Behavioral Data Pipeline"
)
connors = st.Page("connors/connors_app.py", title="Connors 4 Pipeline")

pg = st.navigation(
    [
        homepage,
        bids_conversion,
        participants_tsv,
        participants_demographics,
        add_dosages,
        move_files,
        events,
        behavioral_data,
        connors,
    ],
    position="top",
)
pg.run()
