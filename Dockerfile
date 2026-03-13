# Use a base image (e.g., Ubuntu)
FROM ubuntu:latest

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    sudo \
    software-properties-common \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    build-essential \
    xvfb \
    freeglut3-dev \
    libglm-dev \
    procps \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    python3 \
    python3-pip \
    python3-numpy \
    python3-pandas \
    python3-skimage \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages
RUN pip3 install --break-system-packages skan torch munkres

# Create required symlinks (as root)
RUN if [ -f /usr/lib/x86_64-linux-gnu/libglut.so.3.12 ]; then \
      ln -s /usr/lib/x86_64-linux-gnu/libglut.so.3.12 /usr/lib/x86_64-linux-gnu/libglut.so.3; \
    elif [ -f /usr/lib/x86_64-linux-gnu/libglut.so ]; then \
      ln -s /usr/lib/x86_64-linux-gnu/libglut.so /usr/lib/x86_64-linux-gnu/libglut.so.3; \
    fi

# Create ubuntu user if not exists (ubuntu:latest usually creates uid 1000)
# Ensure correct permissions for /app
COPY . /app
# RUN chown -R 1000:1000 /app

# The container will run as user 1000 (specified in docker-compose)
# But we need to ensure Xvfb can be started properly by user
# Setup environment variables needed for VLAB
ENV DISPLAY=:99

# Specify a default command (optional)
CMD ["bash"]
