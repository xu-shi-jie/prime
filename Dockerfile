ARG FROM_IMAGE_NAME=pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel
FROM ${FROM_IMAGE_NAME} AS prime

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
RUN apt-get update -y \
    && apt-get install -y build-essential python3-dev make apt-utils unzip p7zip-full gpg doxygen git curl mmseqs2 \
    aria2 vim screen rsync wget locales gfortran libglew-dev libpng-dev libfreetype6-dev libxml2-dev libmsgpack-dev \
    python3-pyqt5.qtopengl libglm-dev libnetcdf-dev \
    && locale-gen en_US.UTF-8 \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update -y \
    && apt-get install -y cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace

# install pymol
RUN git clone https://github.com/schrodinger/pymol-open-source.git \
    && git clone https://github.com/rcsb/mmtf-cpp.git \
    && mv mmtf-cpp/include/mmtf* pymol-open-source/include/ \
    && cd pymol-open-source \
    && python3 setup.py build install \
    && cd .. \
    && rm -rf pymol-open-source mmtf-cpp

# install illustrate
RUN git clone https://github.com/ccsb-scripps/Illustrate \
    && cd Illustrate \
    && gfortran illustrate.f -o illustrate \
    && mv illustrate /usr/bin/ \
    && cd .. \
    && rm -rf Illustrate

ADD requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uv && python -m uv pip install -r requirements.txt
ADD . .

RUN git config --global --add safe.directory /workspace

ENV OMP_NUM_THREADS=1
ENV MKL_SERVICE_FORCE_INTEL=TRUE
ENV MKL_THREADING_LAYER=GNU
