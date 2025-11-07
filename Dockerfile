FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

# System deps required by TensorFlow/Scipy and for saving images
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libatlas-base-dev \
        libopenblas-dev \
        liblapack-dev \
        libgl1 \
        libglib2.0-0 \
        graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command runs the training script; override in `docker run` as needed
CMD ["python3", "NoisyTraining.py"]
