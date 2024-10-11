# FROM spark:python3
FROM ubuntu:24.10

USER root

# Installing python, backend for torchaudio, git
RUN set -ex; \
    apt-get update && apt-get upgrade -y; \
    apt-get install -y python3 python3-pip python3-venv ffmpeg git curl

WORKDIR ~/.local/src
RUN git clone https://github.com/dburian/dotfiles; \
    dotfiles/.local/bin/rootless_min_tty_setup.sh; \
    python3 -m venv ~/.venv; \
    ~/.venv/bin/pip install pyright ruff isort git+https://github.com/pre-commit/mirrors-mypy setuptools

WORKDIR /root/youtube_asr

# Install project dependencies
COPY pyproject.toml pyproject.toml
COPY src src
RUN ~/.venv/bin/pip install -e .

RUN echo ". ~/.venv/bin/activate" >> ~/.bashrc
