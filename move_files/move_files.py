import shutil, traceback
from pathlib import Path

from bidsaid.files import get_entity_value
from bidsaid.logging import setup_logger

LGR = setup_logger(__name__)

def run_pipeline(src_dir, bids_dir):
    src_dir = Path(src_dir)
    bids_dir = Path(bids_dir)

    if not src_dir.exists():
        raise FileExistsError(f"The following directory does not exist: {src_dir}")

    for src_file in src_dir.rglob("*"):
        if not src_file.name.startswith("sub-"):
            continue

        subject = get_entity_value(src_file, "sub", return_entity_prefix=True)
        session = get_entity_value(src_file, "ses", return_entity_prefix=True)

        dst_dir = (
            bids_dir
            / subject
            / (session or "")
            / ("" if src_file.name.endswith("sessions.tsv") else "func")
        )

        try:
            dst_file = dst_dir / src_file.name
            if not dst_file.parent.exists():
                continue

            if dst_file.exists():
                dst_file.unlink()

            shutil.move(str(src_file), dst_dir)
        except Exception as e:
            LGR.fatal(traceback.format_exc())
            raise e


