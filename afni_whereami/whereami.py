import requests, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _general_utils import _check_coordinate

CONFIG_PATH = Path(__file__).parent.parent / "config.txt"


def get_server():
    with open(CONFIG_PATH) as f:
        return f.read().strip()


def run_pipeline(mni_coordinate):
    _check_coordinate(mni_coordinate)

    request_data = {
        "x": mni_coordinate[0],
        "y": mni_coordinate[1],
        "z": mni_coordinate[2],
    }
    response = requests.post(f"{get_server()}/", json=request_data)

    return response.json()["output"]
