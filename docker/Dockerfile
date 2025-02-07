# Use the official Ubuntu 22.04 LTS base image
FROM ubuntu:22.04

# Set environment variables to non-interactive
#ENV DEBIAN_FRONTEND=noninteractive

# Update and install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /workspace

# Copy the files from your host machine into the container
COPY ./copy_inside/ /workspace/

# Run scripts
RUN chmod +x /workspace/container_install.sh

RUN ./container_install.sh

# Command to run when the container starts
CMD ["/bin/bash", "-c", "python3 packetCapture.py --filter --websites_count 10000 --access_count 1 --gap_count 10000 &> output.txt && tail -f /dev/null"]
