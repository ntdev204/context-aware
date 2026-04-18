# Context-Aware AI Navigation System 🤖

![CI Status](https://github.com/ntdev204/context-aware/actions/workflows/ci.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/platform-NVIDIA%20Jetson-green.svg)

A high-performance, context-aware artificial intelligence server designed for **Mecanum Robots**. Built specifically for deployment on **NVIDIA Jetson Orin Nano Super**, this system integrates real-time computer vision, human intent prediction, and safety-first navigation.

---

## 🏗 System Architecture

The project is structured into modular components to ensure low-latency processing and reliable communication:

- **Perception Layer**:
  - **YOLO Detection**: Object detection using COCO classes with depth-aware obstacle mapping.
  - **Fallback Tracker**: IoU-based multi-object tracking for stable track identity.
  - **Intent CNN**: Predicts human movement intent (STATIONARY, ERRATIC, TOWARDS, etc.) from extracted ROIs.
- **Decision Layer**:
  - **Context Builder**: Aggregates sensor data, detections, and intent and builds a unified world state.
  - **Navigation Policy**: Heuristic-based navigation with dynamic safety monitoring.
- **Infrastructure**:
  - **ZMQ & Protobuf**: High-speed, language-agnostic messaging between modules.
  - **Experience Buffer**: Efficient logging of real-time experiment data to HDF5 format for academic research.

---

## 🚀 Getting Started

### Prerequisites

- **Deployment**: NVIDIA Jetson Orin Nano Super (L4T)
- **Development**: Docker, NVIDIA Container Toolkit, GitHub CLI

### Quick Start with Docker

```bash
# Build the development image
docker compose -f docker/docker-compose.yml --profile jetson-dev build jetson-dev

# Run the system in production mode
docker compose -f docker/docker-compose.yml --profile jetson-prod up -d
```

### Manual Installation (Development)

```bash
# Install dependencies
pip install -e ".[dev,cpu]"

# Generate Protobuf stubs
python scripts/infra/generate_proto.py

# Run tests
pytest tests/ -v
```

---

## 🧪 Research & Experiments

All experimental data is structured and logged into the `research_data/` directory.

- **Automated Logging**: Real-time metrics and ROI frames are captured during robot runs.
- **Analysis Scripts**: See `scripts/data/` for data exploration and validation tools.
- **Syncing**: Automated background sync to remote research servers via `rsync`.

---

## 🛠 Project Structure

```text
├── .agent/             # AI Agent configuration & skills
├── config/             # Environment-specific YAML configs
├── docker/             # Dockerfiles & Compose configurations
├── proto/              # Protobuf definitions (.proto)
├── research_data/      # Experimental logs and documentation
├── scripts/            # Infrastructure, training, and sync scripts
├── src/                # Core Python source code
└── tests/              # Comprehensive test suite
```

---

## 📄 License & Publication

This project is part of an ongoing research on **Context-Aware Navigation for Mobile Robots**. 

For academic inquiries or citation, please refer to the `research_data/README.md`.

---
*Created with ❤️ by the Context-Aware AI Team.*
