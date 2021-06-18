FROM nvcr.io/nvidia/pytorch:21.05-py3

ENV DEBIAN_FRONTEND="noninteractive" TZ="America/Sao_Paulo"

RUN apt-get update -y && apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig gedit net-tools xauth

RUN pip install 'gym[box2d]' ipywidgets notebook

RUN mkdir -p /home/workspace

WORKDIR /home/workspace