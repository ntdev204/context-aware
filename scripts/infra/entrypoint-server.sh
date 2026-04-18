#!/bin/bash
set -e

# Setup SSH for receiving data from Jetson
mkdir -p /var/run/sshd

# Start SSH daemon
/usr/sbin/sshd

# Prepare volume directories
mkdir -p /data/roi_incoming /data/intent_dataset /workspace/models/cnn_intent

echo "SSH Server started on port 22 (mapped to 2222 on Host)."
echo "Watcher daemon starting..."

# Start the auto watcher
exec python scripts/data/auto_watcher.py
