TAG ?= 2.16.1
IMAGE ?= tensorflow/tensorflow:$(TAG)
GPUNUM ?= 3
DOCKERFLAGS ?= -it --rm -v ./:/sim -w /sim --user $(shell id -u):$(shell id -g)

PLAYS ?= plays/non_iid_20_percent_lost_updates.csv plays/non_iid_all_updates_applied.csv
DATASETS ?= datasets/*.npz
LOGS ?= logs/
HYPERPARAMS ?=
SIMFLAGS ?= -p $(PLAYS) -f $(DATASETS) -l $(LOGS) $(HYPERPARAMS)

run:
	./venv/bin/python ./main.py -p ./plays/play1.csv

docker-runall: logs/
	docker run $(DOCKERFLAGS) $(IMAGE) python main.py $(SIMFLAGS)

docker-runall-gpu: logs/
	docker run $(DOCKERFLAGS) --gpus '"device=$(GPUNUM)"' $(IMAGE)-gpu python main.py $(SIMFLAGS)

tensorboard: logs/
	./venv/bin/tensorboard --logdir logs/

setup: venv/
	./venv/bin/pip install -r requirements.txt

venv/:
	python -m venv ./venv

logs/:
	mkdir -p logs/

clean: reset
	rm -rf venv/

reset:
	rm -f logs/*.log
	rm -f logs/*.done

.PHONY: run docker-runall docker-runall-gpu tensorboard setup clean reset
