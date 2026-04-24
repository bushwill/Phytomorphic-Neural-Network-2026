#!/bin/bash

# 1. Pre-create expected directories as the host user.
# This prevents the Docker Daemon from generating them as 'root' when bind-mounting.
mkdir -p "vlab/oofs/ext/Datasets"
mkdir -p "vlab/oofs/ext/Optimizer Data"
mkdir -p "vlab/oofs/ext/Training Data"

# 2. Fix any existing root-owned folders inside the volume space, safely assigning them back to you.
# (This uses a transient container to run root-level chown without requiring host sudo).
docker run --rm -v "$PWD:/work" -u root ubuntu chown -R $(id -u):$(id -g) /work/vlab/oofs/ext 2>/dev/null || true

# 3. Start the services normally
docker compose up --force-recreate --build -d