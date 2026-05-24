"""
For documentation, a copy of the Python file running fastapi that is hosted on a continuously running instance
on Oracle Cloud. It's the free tier so the instance only has 1 GB ram and 1 OCPU.
"""

import subprocess
from fastapi import FastAPI

ATLAS_NAMES = {
    "kids": "Haskins_Pediatric_Nonlinear_1.01",
    "adults": "FS.afni.MNI2009c_asym",
}

app = FastAPI()


@app.post("/")
def whereami(request_data: dict):
    atlas = ATLAS_NAMES[request_data["cohort"]]
    cmd = (
        f"podman exec afni whereami "
        f"{request_data['x']} {request_data['y']} {request_data['z']} "
        f"-lpi -space MNI -atlas {atlas}"
    )
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    return {"output": output.stdout}
