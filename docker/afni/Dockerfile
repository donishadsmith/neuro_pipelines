# https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/install_instructs/steps_linux_ubuntu24.html

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND="noninteractive"
ENV LANG="en_GB.UTF-8"

RUN apt update -y && \
    apt upgrade -y && \
    apt install software-properties-common -y

RUN add-apt-repository universe && \
    apt upgrade -y && \
    apt install -y \
    tcsh \
    xfonts-base \
    libssl-dev \
    python-is-python3 \
    python3-matplotlib \
    python3-numpy  \
    python3-flask \
    python3-flask-cors \
    python3-pil \
    gsl-bin \
    netpbm \
    gnome-tweaks \
    libjpeg62 \
    xvfb \
    xterm \
    vim \
    curl \
    gedit \
    evince \
    eog \
    libglu1-mesa-dev \
    libglw1-mesa-dev \
    libxm4 \
    build-essential \
    libcurl4-openssl-dev \
    libxml2-dev \
    libgfortran-14-dev \
    libgomp1 \
    gnome-terminal \
    nautilus \
    firefox \
    xfonts-100dpi \
    r-base-dev \
    cmake \
    bc \
    git \
    libgdal-dev \
    libopenblas-dev \
    libnode-dev \
    libudunits2-dev && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

ENV AFNI_DIR="/opt/afni"

# Based on tcsh @update.afni.binaries -help
# -bindir ABIN          : set AFNI binary directory to ABIN
RUN curl -O https://afni.nimh.nih.gov/pub/dist/bin/misc/@update.afni.binaries && \
    tcsh @update.afni.binaries -package linux_ubuntu_24_64 -do_extras -bindir $AFNI_DIR

ENV PATH="$AFNI_DIR:$PATH"
ENV R_LIBS=$AFNI_DIR/R

RUN mkdir $R_LIBS

RUN echo 'export R_LIBS=/opt/afni/R' >> ~/.bashrc && \
    echo 'setenv R_LIBS /opt/afni/R' >> ~/.cshrc

RUN rPkgsInstall -pkgs ALL

RUN useradd -md /home/user user
USER user

CMD ["bash"]
