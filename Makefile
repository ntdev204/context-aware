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
	@echo "  Jetson (ARM64 + CUDA) - Dev Mode"
	@echo "    make jetson-build-dev       Build jetson-dev image"
	@echo "    make jetson-shell           Interactive shell"
	@echo "    make jetson-test            Run tests on Jetson"
	@echo "    make jetson-download-models Download YOLO weights"
	@echo "    make jetson-export          Export YOLO to TensorRT engine"
	@echo "    make jetson-health          RAM / GPU / temp report"
	@echo "    make jetson-bench           GPU FPS benchmark"
	@echo ""
	@echo "  Production"
	@echo "    make jetson-build-prod  Build production image"
	@echo "    make jetson-up          Start server (background)"
	@echo "    make jetson-logs        Tail logs"
	@echo "    make jetson-restart     Restart after update"
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

jetson-build-dev:
	$(JETSON_DEV) build jetson-dev

jetson-shell:
	$(JETSON_DEV) run --rm -it -p 8080:8080 jetson-dev

jetson-test:
	$(JETSON_DEV) run --rm jetson-dev python -m pytest tests/ -v --tb=short

jetson-download-models:
	@mkdir -p models/yolo
	@if [ ! -f models/yolo/yolo11n.pt ]; then \
		echo "Downloading yolo11n.pt (18MB)..."; \
		curl -L -o models/yolo/yolo11n.pt \
			https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt; \
		echo "Downloaded: models/yolo/yolo11n.pt"; \
	else \
		echo "models/yolo/yolo11n.pt already present."; \
	fi

jetson-export: jetson-download-models
	$(JETSON_DEV) run --rm jetson-dev \
		python scripts/deploy/export_engine.py models/yolo/yolo11n.pt --fp16 --workspace 2 --imgsz 320 320
	@echo "TensorRT engine ready: models/yolo/yolo11n.engine"

jetson-proto:
	$(JETSON_DEV) run --rm jetson-dev python scripts/infra/generate_proto.py

jetson-health:
	$(JETSON_DEV) run --rm jetson-dev python scripts/infra/health_check.py

jetson-bench:
	$(JETSON_DEV) run --rm jetson-dev python scripts/deploy/benchmark.py --frames 300

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

jetson-build-prod:
	$(JETSON_PROD) build jetson-prod

jetson-up:
	$(JETSON_PROD) up -d

jetson-logs:
	$(JETSON_PROD) logs -f jetson-prod

jetson-down:
	$(JETSON_PROD) down

jetson-restart:
	$(JETSON_PROD) restart jetson-prod

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
