TAG ?= 2.16.1
IMAGE ?= tensorflow/tensorflow:$(TAG)
GPUNUM ?= 3
DOCKERFLAGS ?= -it --rm -v ./:/sim -w /sim --user $(shell id -u):$(shell id -g)

PLAYS ?= plays/test.csv
LOGS ?= logs/
HYPERPARAMS ?=
SIMFLAGS ?= -p $(PLAYS) -l $(LOGS) $(HYPERPARAMS)

run: logs/
	./venv/bin/python ./main.py -p ./plays/test.csv -l ./logs/

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

clean:
	rm -rf venv/
	rm -rf logs/

reset:
	rm -f logs/*.log
	rm -f logs/*.done

.PHONY: run docker-runall docker-runall-gpu tensorboard setup clean reset
