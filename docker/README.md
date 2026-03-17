# Docker

To install using docker, ensure docker is [installed](https://docs.docker.com/engine/install/) and use one of the following commands in your preferred terminal (Note: change backward to forward slashes on Linux or Mac):

If in the same directory as the Dockerfile:

```bash
docker build -t {image_name} .
```

If building from the root directory:

For AFNI:
```bash
docker build -t afni -f  .\docker\afni\Dockerfile .
```

For FSL with Octave + Palm:
```bash
docker build -t fsl -f  .\docker\fsl\Dockerfile .
```

# HPC

To build an Apptainer/Singularity image on the HPC:

For AFNI:
```bash
apptainer build afni.simg docker://donishadsmith/afni-with-r:26.0.09
```

For FSL with Octave + Palm:
```bash
apptainer build fsl.simg docker://donishadsmith/fsl-palm:0.0.1
```
