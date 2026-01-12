# Use a base image (e.g., Ubuntu)
FROM ubuntu:latest

# Set the working directory inside the container
WORKDIR /app

# Copy directory into the container
COPY . /app

# Specify a default command (optional)
CMD ["bash"]
