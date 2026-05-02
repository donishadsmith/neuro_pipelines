import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "move_files"))

import streamlit as st

from _streamlit_utils import _select_content

from move_files import run_pipeline

st.set_page_config(layout="centered")

st.title("Move BIDS Files Pipeline")
st.divider()

st.markdown("""**Notes:**\n
- Source files must follow BIDS naming conventions (i.e., filenames must start with ``sub-``).
- Destination subdirectories must already exist - files are skipped if the target directory is not found.
- If a file with the same name already exists in the destination, it is overwritten.""")

st.divider()
st.markdown("**Required Arguments**")

if st.button(
    "Browse for source directory",
    help="Directory containing BIDS-compliant files.",
):
    folder = _select_content("directory")
    if folder:
        st.session_state.origin_dir = folder
        st.session_state.bids_files = sorted(
            [
                x
                for x in Path(st.session_state.origin_dir).glob("*")
                if x.is_file() and re.findall(r"sub-\d{5}", x.name)
            ]
        )

if st.session_state.get("origin_dir"):
    if st.session_state.bids_files:
        st.success(f"Source directory: {st.session_state.origin_dir}")
    else:
        st.error(
            f"Not valid BIDS files detected in source directory: {st.session_state.origin_dir}"
        )

if st.button(
    "Browse for BIDS directory",
    help="The BIDS directory to move the BIDS-compliant to.",
):
    folder = _select_content("directory")
    if folder:
        st.session_state.bids_dir = folder
        st.session_state.bids_subfolders = sorted(
            [
                x
                for x in Path(st.session_state.bids_dir).glob("*")
                if x.is_dir() and re.findall(r"sub-\d{5}", x.name)
            ]
        )

if st.session_state.get("bids_dir"):
    if st.session_state.bids_subfolders:
        st.success(f"BIDS directory: {st.session_state.bids_dir}")
    else:
        st.error(
            f"Not a valid BIDS directory (no subjects detected): {st.session_state.bids_dir}"
        )

kwargs = {
    "origin_dir": st.session_state.get("origin_dir"),
    "bids_dir": st.session_state.get("bids_dir"),
}

st.divider()
if st.button("Run Pipeline", type="primary"):
    if not (st.session_state.get("origin_dir") and st.session_state.get("bids_files")):
        st.error(
            "Please select a valid source directory with BIDS-compliant files before running."
        )
    elif not (
        st.session_state.get("bids_dir") and st.session_state.get("bids_subfolders")
    ):
        st.error("Please select a valid BIDS directory before running.")
    else:
        with st.spinner("Processing..."):
            dst_dir = run_pipeline(**kwargs)

        st.success(
            "Files moved to thier respective subject subdirectories "
            f"at the following BIDS parent directory: {st.session_state.get('bids_dir')}"
        )
