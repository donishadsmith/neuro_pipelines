"""
For documentation: a copy of the Python file running fastapi that is hosted on a continuously running instance
on Oracle Cloud. It's the free tier so the instance only has 1 GB ram and 1 OCPU (VM.Standard.E2.1.Micro)

https://www.freedesktop.org/software/systemd/man/latest/loginctl.html
https://man7.org/linux/man-pages/man2/fallocate.2.html
https://man7.org/linux/man-pages/man8/swapon.8.html

Uses systemd to create and manage a process that runs uvicorn in the background so it can listen to incoming
requests continuously on the cloud vm:
https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html?__goaway_challenge=meta-refresh&__goaway_id=8887fcae4814db9e37888fc15975b6b4&__goaway_referer=https%3A%2F%2Fwww.google.com%2F

https://github.com/fastapi/fastapi/discussions/14783
"""

import subprocess
from fastapi import FastAPI

app = FastAPI()


@app.post("/")
def whereami(request_data: dict):
    cmd = (
        f"podman run localhost/afni whereami "
        f"{request_data['x']} {request_data['y']} {request_data['z']} "
        f"-lpi -space MNI_2009c_asym -atlas FS.afni.MNI2009c_asym -atlas Brodmann_Pijn_AFNI"
    )
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    return {"output": output.stdout}
