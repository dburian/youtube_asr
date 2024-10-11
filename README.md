# ASR task from Seznam

This is a repo containing my solution to the given task. The description of my
solution is in `report/report.pdf`

## Installation

The installation should be straightforward as it is a docker image. Run:

```bash
docker compose up jupyter
```
to get a Jupyter notebook running. You might want to look into the compose file,
to specify some variables like jupyter lab port, tensorflow port and cache
directories.

### Data

Download and place the data under `./data/`.

## Project structure

This project contains several notebooks under `./notebooks` and a package called
`youtube_asr` under `./src/youtube_asr`. The package contains some more involved
code or code that is shared between the notebooks. The notebooks showcase my
findings and results.

There is also a logging directory under `./logs/logs`. Fire-up Tensorboard to
look at logs of various experiments I did.
