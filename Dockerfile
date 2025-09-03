# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    unzip \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgcc-s1 \
    ca-certificates \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /app

# Clone RT-DETRv2 repository
RUN git clone https://github.com/lyuwenyu/RT-DETR.git
WORKDIR /app/RT-DETR/rtdetrv2_pytorch

# Install RT-DETR requirements
RUN pip3 install -r requirements.txt

# Set Python path
ENV PYTHONPATH=/app/rtdetrv2_pytorch:$PYTHONPATH

# Go back to main app directory
WORKDIR /app

# Copy project files
COPY . /app/

# Default command
# CMD ["bash"]
