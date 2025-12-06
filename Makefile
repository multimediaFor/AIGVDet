.PHONY: help build-gpu build-cpu build-all run-gpu run-cpu up-gpu up-cpu shell-gpu shell-cpu clean

help:
	@echo "AIGVDet Docker Makefile"
	@echo ""
	@echo "Build commands:"
	@echo "  make build-gpu      - Build GPU Docker image"
	@echo "  make build-cpu      - Build CPU Docker image"
	@echo "  make build-all      - Build both images"
	@echo ""
	@echo "Run commands:"
	@echo "  make up-gpu         - Start GPU container with docker-compose"
	@echo "  make up-cpu         - Start CPU container with docker-compose"
	@echo "  make run-gpu        - Run GPU container interactively"
	@echo "  make run-cpu        - Run CPU container interactively"
	@echo ""
	@echo "Development:"
	@echo "  make shell-gpu      - Open bash shell in GPU container"
	@echo "  make shell-cpu      - Open bash shell in CPU container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove containers and images"

build-gpu:
	docker build -f Dockerfile.gpu -t aigvdet:gpu .

build-cpu:
	docker build -f Dockerfile.cpu -t aigvdet:cpu .

build-all: build-cpu build-gpu

run-gpu:
	docker run --gpus all -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/checkpoints:/app/checkpoints \
		-v $(PWD)/raft_model:/app/raft_model \
		-v $(PWD)/logs:/app/logs \
		-p 6006:6006 \
		aigvdet:gpu

run-cpu:
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/checkpoints:/app/checkpoints \
		-v $(PWD)/raft_model:/app/raft_model \
		-v $(PWD)/logs:/app/logs \
		-p 6006:6006 \
		aigvdet:cpu

up-gpu:
	docker-compose up aigvdet-gpu

up-cpu:
	docker-compose up aigvdet-cpu

shell-gpu:
	docker run --gpus all -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/checkpoints:/app/checkpoints \
		-v $(PWD)/raft_model:/app/raft_model \
		aigvdet:gpu /bin/bash

shell-cpu:
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/checkpoints:/app/checkpoints \
		aigvdet:cpu /bin/bash

clean:
	docker-compose down
	docker rmi aigvdet:gpu aigvdet:cpu || true

train-gpu:
	docker-compose run aigvdet-gpu python3.11 train.py --gpus 0 --exp_name default_exp

train-cpu:
	docker-compose run aigvdet-cpu python train.py --exp_name default_exp

tensorboard:
	docker run --gpus all -it --rm \
		-v $(PWD)/logs:/app/logs \
		-p 6006:6006 \
		aigvdet:gpu \
		tensorboard --logdir=/app/logs --host=0.0.0.0 --port=6006
