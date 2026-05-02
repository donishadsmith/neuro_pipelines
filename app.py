import streamlit as st

homepage = st.Page("_homepage.py", title="Home")
subjects_visits = st.Page(
    "bids_conversion_pipeline/streamlit/subjects_visits_app.py",
    title="Subjects Visits File Pipeline",
)
bids_conversion = st.Page(
    "bids_conversion_pipeline/streamlit/bids_conversion_app.py",
    title="NIfTI to BIDS Pipeline",
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
events = st.Page("events/bids_events_app.py", title="BIDS Events Pipeline")
behavioral_data = st.Page(
    "events/behavioral_data_app.py", title="Behavioral Data Pipeline"
)
move_files = st.Page("move_files/move_files_app.py", title="Move BIDS Files Pipeline")
connors = st.Page("connors/connors_app.py", title="Connors 4 Pipeline")
nih_toolbox = st.Page("nih_toolbox/nih_toolbox_app.py", title="NIH Toolbox Pipeline")

pg = st.navigation(
    [
        homepage,
        subjects_visits,
        bids_conversion,
        participants_tsv,
        participants_demographics,
        add_dosages,
        events,
        behavioral_data,
        move_files,
        connors,
        nih_toolbox,
    ],
    position="sidebar",
)
pg.run()
