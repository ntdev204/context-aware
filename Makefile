COMPOSE     = docker compose -f docker/docker-compose.yml
DEV         = $(COMPOSE) --profile dev
JETSON_DEV  = $(COMPOSE) --profile jetson-dev
JETSON_PROD  = $(COMPOSE) --profile jetson-prod
SERVER_TRAIN = $(COMPOSE) --profile server

# Sync config is now located entirely in env/sync.env and handled by Docker

.PHONY: help \
        jetson-build-dev jetson-shell jetson-test jetson-export \
        jetson-download-models jetson-proto jetson-health jetson-bench \
        jetson-build-prod jetson-up jetson-logs jetson-down jetson-restart \
        server-build server-up server-logs server-down server-shell \
        server-explore server-train-now \
        clean clean-all clean-images

# Default
help:
	@echo ""
	@echo "  Context-Aware AI Server -- Makefile"
	@echo ""
	@echo "  Jetson Native (ARM64 + CUDA) - No Docker"
	@echo "    make jetson-setup           Install native Jetson dependencies"
	@echo "    make jetson-test            Run Pytest natively"
	@echo "    make jetson-download-models Download YOLO weights"
	@echo "    make jetson-export          Export YOLO to TensorRT engine natively"
	@echo "    make jetson-health          RAM / GPU / temp report"
	@echo "    make jetson-bench           GPU FPS benchmark"
	@echo ""
	@echo "  Jetson Native Production"
	@echo "    make jetson-up          Start server (background nohup)"
	@echo "    make jetson-logs        Tail native logs"
	@echo "    make jetson-restart     Restart native daemon"
	@echo "    make jetson-down        Stop server"
	@echo ""
	@echo "  Training Server"
	@echo "    make server-build       Build training server image"
	@echo "    make server-up          Start training server daemon"
	@echo "    make server-logs        Tail watcher daemon logs"
	@echo "    make server-explore     Run data exploration on dataset"
	@echo "    make server-train-now   Trigger manual training run"
	@echo "    make server-shell       Interactive shell"
	@echo "    make server-down        Stop training server"
	@echo ""
	@echo "  Cleanup"
	@echo "    make clean              Remove stopped containers"
	@echo "    make clean-images       Remove dangling images"
	@echo "    make clean-all          Remove images + volumes"
	@echo ""


# Dev (CPU) section removed.

# Jetson Dev

jetson-setup:
	@echo "Setting up Jetson native environment..."
	sudo apt-get update && sudo apt-get install -y python3-pip libgl1 libglib2.0-0
	pip3 install -r requirements.txt ultralytics==8.3.0

jetson-test:
	python3 -m pytest tests/ -v --tb=short

jetson-download-models:
	@mkdir -p models/yolo
	@if [ ! -f models/yolo/yolo11s.pt ]; then \
		echo "Downloading yolo11s.pt (18MB)..."; \
		curl -L -o models/yolo/yolo11s.pt \
			https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt; \
		echo "Downloaded: models/yolo/yolo11s.pt"; \
	else \
		echo "models/yolo/yolo11s.pt already present."; \
	fi

jetson-export: jetson-download-models
	python3 scripts/deploy/export_engine.py models/yolo/yolo11s.pt --fp16 --workspace 1 --imgsz 480 640
	@echo "TensorRT engine ready: models/yolo/yolo11s.engine"

jetson-proto:
	python3 scripts/infra/generate_proto.py

jetson-health:
	python3 scripts/infra/health_check.py

jetson-bench:
	python3 scripts/deploy/benchmark.py --frames 300

# Training Server

server-build:
	$(SERVER_TRAIN) build server-train

server-up:
	$(SERVER_TRAIN) up -d server-train

server-logs:
	$(SERVER_TRAIN) logs -f server-train

server-down:
	$(SERVER_TRAIN) down

server-shell:
	$(SERVER_TRAIN) run --rm -it server-train bash

server-explore:
	$(SERVER_TRAIN) run --rm server-train python scripts/data/explore_roi.py /data/intent_dataset

server-train-now:
	$(SERVER_TRAIN) run --rm server-train python scripts/train/train_intent_cnn.py \
		--dataset /data/intent_dataset --epochs 10

# Production

jetson-up:
	@echo "Starting Jetson native daemon..."
	nohup python3 scripts/core/main.py > jetson.log 2>&1 & echo $$! > jetson.pid
	@echo "Daemon started. PID saved to jetson.pid"

jetson-logs:
	tail -f jetson.log

jetson-down:
	@if [ -f jetson.pid ]; then \
		kill -9 `cat jetson.pid` || true; \
		rm jetson.pid; \
		echo "Daemon stopped."; \
	else \
		echo "Daemon not running."; \
	fi

jetson-restart: jetson-down jetson-up

# Cleanup

clean:
	$(COMPOSE) --profile jetson-dev --profile jetson-prod \
		down --remove-orphans

clean-images:
	docker image prune -f

clean-all:
	$(COMPOSE) --profile jetson-dev --profile jetson-prod \
		down --remove-orphans --volumes
	docker rmi context-aware:jetson-dev \
		context-aware:jetson-prod 2>/dev/null || true
