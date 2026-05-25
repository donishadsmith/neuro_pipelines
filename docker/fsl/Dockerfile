FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_GB.UTF-8

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y \
    python3 \
    wget \
    unzip \
    file \
    dc \
    libquadmath0 \
    libgomp1 \
    octave \
    octave-image \
    octave-statistics && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py && \
    python fslinstaller.py -d /usr/local/fsl/ && \
    rm fslinstaller.py

RUN wget https://github.com/andersonwinkler/PALM/archive/master.zip && \
    unzip master.zip -d /opt/ && \
    rm master.zip && \
    mv /opt/PALM-master /opt/palm

RUN echo "function p = fsgettmppath; p = tempdir; end" > /opt/palm/fsgettmppath.m

RUN echo "pkg load image; pkg load statistics;" >> /usr/share/octave/site/m/startup/octaverc && \
    echo "addpath(genpath('/opt/palm'));" >> /usr/share/octave/site/m/startup/octaverc && \
    echo "source /usr/local/fsl/etc/fslconf/fsl.sh" > /etc/profile.d/fsl.sh

ENV FSLDIR=/usr/local/fsl
ENV PATH=$FSLDIR/bin:$PATH
ENV FSLOUTPUTTYPE=NIFTI_GZ

CMD ["bash"]
