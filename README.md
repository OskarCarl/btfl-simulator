# Simulator for BitTorrent based Gossip Learning

This is a simulator built to evaluate and analyse the idea
to structure gossip-based machine learning
using BitTorrent.
It is part of a submission to the poster track
of the 14th International Conference on the Internet of Things by the ACM.

The plays and log files used in the evaluation in the paper
can be found in [the evaluation folder](evaluation/).

## Reproducing the Simulations

The easiest way is to use Make and Docker.
To reproduce these results you can run the plays like this:

```sh
## Scenario 1
PLAYS="evaluation/scenario_1.csv" HYPERPARAMS="--numpeers 30 --numneighbours 4 --retrainfactor 0.5" make docker-runall-gpu

## Scenario 2
PLAYS="evaluation/scenario_2.csv" HYPERPARAMS="--numpeers 30 --numneighbours 2 --retrainfactor 0.5" make docker-runall-gpu

## Scenario 2 (alternative)
PLAYS="evaluation/scenario_2.csv" HYPERPARAMS="--numpeers 30 --numneighbours 4 --retrainfactor 0.5" make docker-runall-gpu

## Reference
PLAYS="evaluation/reference.csv" HYPERPARAMS="--numpeers 1" make docker-runall-gpu
```

This will take a while and write logs to `logs/<timestamp>_<play-name>.log`.
Progress is printed to stdout.
Once a play is done, an identically named `.done` file is created.

For the alternate of scenario 1 the peer selection strategy needs to be changed in [the peer source code](peer/peer.py#L37).

Please be aware that results might differ slightly
since the peer selection includes randomized ordering for peers with equal model age.

If the execution should now be done on a GPU, replace the make target with `docker-runall`

## Reproducing the Analysis

This uses a Python script based on matplotlib and numpy.
To install the dependencies, run `make setup`.
The script can then be run like this:

```sh
venv/bin/python scripts/scatterplot.py evaluation/scenario_1.csv evaluation/reference.csv
```

The first argument is the log file of the simulation,
the second argument is the log file of the reference simulation.
The script will output a PDF file to the current directory
and open an interactive GUI in the browser.

To include the zoomed insets,
the [relevant code](scripts/scatterplot.py#L58-L83) needs to be uncommented in the script.
The insets for scenario 1 and 2 require different blocks of code,
please see the comments in the script.

The script also prints some statistics
about the scenario (not the reference) to stdout.
