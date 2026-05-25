"""
For documentation, a copy of the Python file running fastapi that is hosted on a continuously running instance
on Oracle Cloud. It's the free tier so the instance only has 1 GB ram and 1 OCPU.
"""

import subprocess
from fastapi import FastAPI

app = FastAPI()


@app.post("/")
def whereami(request_data: dict):
    cmd = (
        f"podman exec afni whereami "
        f"{request_data['x']} {request_data['y']} {request_data['z']} "
        f"-lpi -space MNI_2009c_asym -atlas FS.afni.MNI2009c_asym -atlas Brodmann_Pijn_AFNI"
    )
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    return {"output": output.stdout}
