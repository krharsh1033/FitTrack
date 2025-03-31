# Use an ARM64-compatible base image
FROM python:3.11-slim

# Set environment variables to avoid interactive prompts and ensure non-root user setup
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    CMAKE_ARGS="-DLLAMA_NATIVE=ON"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install and upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .

# Try installing llama-cpp-python with prebuilt wheels first (if available)
RUN pip install --no-cache-dir --prefer-binary --force-reinstall llama-cpp-python

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set default command
CMD ["python", "app.py"]
