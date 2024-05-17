# goal-simulation

A python simulation toolkit for testing goal/intent recognition algorithms in mobility study.

- Based on a widely used Agent-based modeling platform: [mesa](https://mesa.readthedocs.io/en/stable/).
- Has the ability to work with scenes including grid space, network space and continuous Euclidean space.
- Includes a few path planning algorithms from robotics for movement generation.
- Implemented a few well-known goal recognition algorithms as benchmarks.
- Visual simulation available for grid space.

----
## Installation
```console
$ conda create -n intention_recognition -file requirements.txt
$ conda activate intention_recognition
```

----
## Example use cases
1. Run paper experiments in batch
Please see paper_experiments.ipynb as an example 

2. Visual simulation with the command
```console
$ mesa runserver
```
The parameters for the simulation can be changed in server.py

