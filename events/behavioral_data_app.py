import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "events"))

from _events_utils import _app

from get_behavioral_data import run_pipeline

_app(caller="Behavioral Data", pipeline=run_pipeline)
