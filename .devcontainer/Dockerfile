# Use the official Python 3.12 image from the Microsoft Container Registry
FROM mcr.microsoft.com/devcontainers/python:3.12

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    curl \
    git \
    make \
    pandoc \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspaces/ai-document-analyzer

# Copy project files into the container (optional, devcontainers usually mount)
# COPY . .

# Install Python dependencies
# Note: We'll usually let the postCreateCommand handle this to ensure it's in the right environment
# RUN pip install --upgrade pip && pip install -e ".[dev,sphinx]"
