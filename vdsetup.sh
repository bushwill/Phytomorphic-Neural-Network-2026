#!/bin/bash

apt-get update && apt-get install -y sudo

sudo apt-get update && \
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
sudo add-apt-repository -y universe && \
sudo apt-get install -y qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools build-essential freeglut3-dev
sudo apt-get install -y xvfb x11vnc
sudo apt-get install -y lxde
sudo apt-get install -y python3 python3-pip python3-numpy python3-pandas python3-skimage python3-munkres
sudo pip3 install --break-system-packages skan torch
sudo git clone https://github.com/novnc/noVNC.git /opt/noVNC

ln -s /usr/lib/x86_64-linux-gnu/libglut.so.3.12 /usr/lib/x86_64-linux-gnu/libglut.so.3

# Initialize virtual display for vlab
Xvfb :99 -screen 0 1280x1024x24 2>/dev/null &
# Start x11vnc server
x11vnc -display :99 -rfbport 6079 -forever -shared -bg
# Start noVNC server
cat > /opt/noVNC/defaults.json << EOF
{
    "resize": "scale",
}
EOF
/opt/noVNC/utils/novnc_proxy --vnc localhost:6079 --listen 6080 &
# Create a symbolic link for noVNC HTML file
ln -sf /opt/noVNC/vnc.html /opt/noVNC/index.html

# Set environment variables for the virtual display
export DISPLAY=:99
# Start LXDE desktop environment
startlxde &
# Initialize the runtime directory for XDG
export XDG_RUNTIME_DIR=/tmp/runtime-root

# Start browser and source the environment setup script
./bin/govlab.sh
# Run environment setup script
. ./bin/sourceme.sh
cd oofs/ext/bushell/nn
