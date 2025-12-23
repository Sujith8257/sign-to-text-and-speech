# Docker Quick Reference Guide

Quick reference for Docker commands and deployment.

## Quick Start

```bash
# Build and run with Docker Compose (easiest)
docker-compose up --build

# Or build and run manually
docker build -t sign-to-text-speech .
docker run -d -p 5000:5000 --name signapp sign-to-text-speech
```

## Common Commands

### Build Images

```bash
# CPU version (default)
docker build -t sign-to-text-speech .

# GPU version (requires NVIDIA Container Toolkit)
docker build -f Dockerfile.gpu -t sign-to-text-speech:gpu .
```

### Run Containers

```bash
# Basic run
docker run -d -p 5000:5000 --name signapp sign-to-text-speech

# With webcam (Linux)
docker run --device /dev/video0:/dev/video0 -p 5000:5000 --name signapp sign-to-text-speech

# With GPU
docker run --gpus all -d -p 5000:5000 --name signapp-gpu sign-to-text-speech:gpu

# With volume mounts (for easy model updates)
docker run -d -p 5000:5000 \
  -v $(pwd)/model.p:/app/model.p:ro \
  -v $(pwd)/indian_sign_model.h5:/app/indian_sign_model.h5:ro \
  --name signapp sign-to-text-speech
```

### Container Management

```bash
# View logs
docker logs -f signapp

# Stop container
docker stop signapp

# Start container
docker start signapp

# Remove container
docker rm signapp

# Remove image
docker rmi sign-to-text-speech
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build -d

# Stop and remove volumes
docker-compose down -v
```

## Troubleshooting

### Check Container Status
```bash
docker ps -a
docker inspect signapp
```

### Access Container Shell
```bash
docker exec -it signapp /bin/bash
```

### View Resource Usage
```bash
docker stats signapp
```

### Clean Up
```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove everything (careful!)
docker system prune -a
```

## Production Deployment

### Environment Variables
```bash
docker run -d -p 5000:5000 \
  -e FLASK_ENV=production \
  -e SECRET_KEY=your-secret-key \
  --name signapp sign-to-text-speech
```

### With Nginx Reverse Proxy
```nginx
upstream signapp {
    server localhost:5000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://signapp;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## GPU Setup

### Install NVIDIA Container Toolkit

**Ubuntu/Debian:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Verify GPU Access:**
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## File Structure

```
.
├── Dockerfile              # CPU Docker image
├── Dockerfile.gpu          # GPU Docker image
├── docker-compose.yml      # Docker Compose config
├── .dockerignore          # Files to exclude from build
└── DOCKER.md              # This file
```
