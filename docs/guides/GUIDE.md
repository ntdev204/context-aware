# Deployment and Usage Guide

This guide covers everything needed to deploy and operate the Context-Aware AI Server on a Jetson Orin Nano Super from scratch.

---

## Table of Contents

1. [Hardware Setup](#hardware-setup)
2. [JetPack Configuration](#jetpack-configuration)
3. [First-Time Docker Deployment](#first-time-docker-deployment)
4. [Model Export](#model-export)
5. [Running the Server](#running-the-server)
6. [Monitoring via Edge API](#monitoring-via-edge-api)
7. [Runtime Control](#runtime-control)
8. [Configuration Tuning](#configuration-tuning)
9. [Data Collection](#data-collection)
10. [Troubleshooting](#troubleshooting)

---

## Hardware Setup

### Camera Wiring

The Orbbec Astra S uses USB 2.0. Connect it directly to a Jetson USB port (not a hub).

Verify the device is recognized on the host:

```bash
lsusb | grep 2bc5
# Expected: Bus 001 Device 005: ID 2bc5:0402 Orbbec 3D Technology International, Inc ASTRA S
```

If the device is not listed, try a different USB port or cable.

### Network

The Jetson communicates with the Raspberry Pi over LAN. Ethernet is preferred; WiFi adds ~2ms latency. Set a static IP for both devices or use a router with static DHCP leases.

| Device | Role | Default IP |
|--------|------|-----------|
| Jetson | AI Server | `100.104.204.128` |
| RasPi 4 | Navigation controller | `25.12.4.101` |

Update `communication.zmq.rasp_pi_ip` in `config/production.yaml` to match your RasPi IP.

---

## JetPack Configuration

Run these once on the Jetson host (not in Docker):

```bash
# Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Disable GUI -- saves ~800MB RAM (reboot required)
sudo systemctl set-default multi-user.target

# Add 16GB swap on NVMe (required for TensorRT engine export)
sudo fallocate -l 16G /mnt/16GB.swap
sudo chmod 600 /mnt/16GB.swap
sudo mkswap /mnt/16GB.swap
sudo swapon /mnt/16GB.swap
echo '/mnt/16GB.swap none swap sw 0 0' | sudo tee -a /etc/fstab

sudo reboot
```

---

## First-Time Docker Deployment

```bash
cd ~/context-aware

# Step 1: Build the Jetson dev image (downloads base image ~5GB first time)
sudo make jetson-build-dev

# Step 2: Download YOLO weights and export TensorRT engine (one-time, ~10 min)
sudo make jetson-export
# This creates: models/yolo/yolo11s.engine

# Step 3: Build production image
sudo make jetson-build-prod

# Step 4: Start in production mode (auto-restart on crash)
sudo make jetson-up

# Step 5: Verify logs
sudo make jetson-logs
```

Expected startup output:

```
INFO     Context-Aware AI Server starting
INFO     Temporal Intent CNN loaded [device=cuda]
INFO     Astra S OpenNI2 streams opened (640x480)
INFO     Camera started: 640x480 @ 30 FPS [astra, depth=True]
INFO     ZMQ Publisher started
INFO     ZMQ Subscriber connected to 25.12.4.101:5560
INFO     Edge API started: http://0.0.0.0:8080  (docs: /docs)
INFO     All components started -- entering inference loop
```

---

## Model Export

The TensorRT engine must match the YOLO input size configured in `production.yaml`.

```bash
# Export with default 480x640 resolution (matches 640x480 camera)
sudo make jetson-export

# Export with custom resolution (H W)
sudo make jetson-shell
python scripts/export_engine.py models/yolo/yolo11s.pt --fp16 --imgsz 480 640
```

The engine is saved to `models/yolo/yolo11s.engine` and persists across container restarts via the Docker volume.

If you change `perception.yolo.input_size` in the config, you must re-export the engine with matching `--imgsz`.

---

## Running the Server

### Production (auto-restart)

```bash
sudo make jetson-up      # start
sudo make jetson-logs    # tail logs
sudo make jetson-down    # stop
sudo make jetson-restart # restart after config or model update
```

### Interactive (development / debugging)

```bash
sudo make jetson-shell
# Inside container:
MODE=production python -m src.main
```

### Development mode (laptop, USB camera)

```bash
make dev-build
make dev-run
```

---

## Monitoring via Edge API

Open the interactive API docs in a browser:

```
http://<jetson-ip>:8080/docs
```

### Check server health

```bash
curl http://100.104.204.128:8080/health
```

```json
{
  "status": "ok",
  "uptime_s": 123.4,
  "fps": 29.8,
  "mode": "CRUISE",
  "mode_override": null
}
```

### Full metrics

```bash
curl http://100.104.204.128:8080/metrics
```

### Live video stream

Open in browser or VLC:

```
http://100.104.204.128:8080/stream
```

VLC:

```bash
vlc http://100.104.204.128:8080/stream
```

### Live metrics via WebSocket

```javascript
const ws = new WebSocket("ws://100.104.204.128:8080/ws/metrics");
ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  console.log(`FPS: ${data.fps}  Mode: ${data.mode}  Persons: ${data.persons}`);
};
```

### Live detection feed via WebSocket

```javascript
const ws = new WebSocket("ws://100.104.204.128:8080/ws/detections");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

---

## Runtime Control

Control the robot from any HTTP client without restarting the server.

### Stop the robot

```bash
curl -X POST http://100.104.204.128:8080/control/stop
```

### Set mode override

Valid modes: `STOP`, `CRUISE`, `CAUTIOUS`, `AVOID`

```bash
curl -X POST http://100.104.204.128:8080/control/mode/CAUTIOUS
```

### Clear override (restore policy)

```bash
curl -X DELETE http://100.104.204.128:8080/control/mode
```

---

## Configuration Tuning

### Update FPS target at runtime

```bash
curl -X PATCH http://25.12.4.100:8080/config \
  -H "Content-Type: application/json" \
  -d '{"fps_target": 20}'
```

### Adjust YOLO confidence threshold at runtime

```bash
curl -X PATCH http://25.12.4.100:8080/config \
  -H "Content-Type: application/json" \
  -d '{"yolo_confidence_threshold": 0.6}'
```

Note: runtime config updates apply to newly created config lookups. Parameters passed directly at startup (e.g., YOLO model path) cannot be changed without a restart.

### Reduce watchdog log rate

If no RasPi is connected, the watchdog will fire every 500ms but only log at the configured interval:

```yaml
navigation:
  safety:
    watchdog_log_interval_s: 10.0  # log at most once per 10 seconds
```

---

## Data Collection

Experience data (frames, detections, commands, robot state) is written to HDF5 files.

### Location

```
logs/experience/<session-id>_<timestamp>.h5
```

### Copying data from Jetson to laptop

```bash
# From laptop
rsync -av rai@<jetson-ip>:~/context-aware/logs/experience/ ./data/raw/
```

### Disable data logging (save I/O)

```yaml
experience:
  enabled: false
```

---

## Troubleshooting

### Camera not detected

```bash
# On Jetson host
lsusb | grep 2bc5

# If not listed: reconnect USB, try different port
# If listed but camera fails in container: verify Docker has USB access
ls /dev/bus/usb/001/
```

The Docker container uses `privileged: true` and mounts `/dev/bus/usb`. If the Astra S appears in `lsusb` on the host but not in the container, check the compose file.

### TensorRT engine mismatch

Error: `Supplied binding dimension [1,3,640,640] exceed min~max range`

The engine was exported with a different input size than configured. Re-export:

```bash
sudo make jetson-export
```

Ensure `perception.yolo.input_size` in `production.yaml` matches the `--imgsz` used during export.

### Watchdog STOP (no RasPi connected)

```
WARNING  Watchdog timeout (500 ms) -- STOP
```

This is expected when no RasPi is sending robot state. The robot will not move. To suppress log spam:

```yaml
navigation:
  safety:
    watchdog_log_interval_s: 30.0
```

Or connect the RasPi and start the ZMQ publisher from the navigation stack.

### High RAM usage

```bash
# Check RAM inside container
sudo make jetson-shell
python scripts/health_check.py
```

If RAM exceeds 3.5GB, reduce batch sizes:

```yaml
perception:
  cnn_intent:
    max_batch_size: 2
experience:
  buffer_size: 5000
```

### Build fails (pip timeout)

The base image overrides PIP_INDEX_URL to a local Jetson server not available during build. This is handled in the Dockerfile. If you see `Connection refused` during pip:

```bash
# Rebuild without cache to re-apply the PIP_INDEX_URL fix
docker compose -f docker/docker-compose.yml --profile jetson-dev build --no-cache jetson-dev
```

### ONNX references in code

ONNX has been removed from this project. Direct `.pt` to `.engine` export is used via Ultralytics. If you see `import onnx` errors, you have a stale Python environment. Rebuild the container.
