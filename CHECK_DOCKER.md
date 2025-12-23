# How to Check if Docker is Working

This guide provides multiple ways to verify that Docker is installed and working correctly on your system.

## Quick Check Methods

### Method 1: Using the Check Script (Easiest)

Run the provided Python script:

```bash
python check_docker.py
```

This will automatically check:
- ✅ Docker installation
- ✅ Docker daemon status
- ✅ Docker Compose availability
- ✅ Docker functionality test
- ✅ System information

### Method 2: Manual Command Line Checks

#### Check Docker Version
```bash
docker --version
```
**Expected output:** `Docker version XX.XX.X, build xxxxx`

#### Check Docker Daemon Status
```bash
docker info
```
**Expected output:** Detailed Docker system information

**Quick status check:**
```bash
docker ps
```
**Expected output:** List of running containers (may be empty if no containers running)

#### Test Docker with Hello World
```bash
docker run --rm hello-world
```
**Expected output:** 
```
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

#### Check Docker Compose
```bash
docker-compose --version
```
or (newer Docker versions):
```bash
docker compose version
```
**Expected output:** `Docker Compose version vX.X.X`

## Platform-Specific Checks

### Windows

1. **Check Docker Desktop Status:**
   - Look for Docker Desktop icon in system tray
   - Right-click → Check if "Docker Desktop is running"

2. **PowerShell Commands:**
   ```powershell
   # Check if Docker service is running
   Get-Service docker
   
   # Check Docker version
   docker --version
   
   # Test Docker
   docker run hello-world
   ```

### Linux

1. **Check Docker Service:**
   ```bash
   sudo systemctl status docker
   ```

2. **Start Docker Service (if not running):**
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker  # Enable on boot
   ```

3. **Check Docker Group:**
   ```bash
   groups  # Should include 'docker'
   ```

### macOS

1. **Check Docker Desktop:**
   - Open Docker Desktop application
   - Check status in menu bar

2. **Terminal Commands:**
   ```bash
   docker --version
   docker info
   ```

## Troubleshooting

### Issue: "docker: command not found"

**Solution:**
- Docker is not installed
- Install Docker Desktop from: https://www.docker.com/get-started
- Or install Docker Engine for Linux

### Issue: "Cannot connect to the Docker daemon"

**Possible causes and solutions:**

1. **Docker Desktop not running (Windows/Mac):**
   - Start Docker Desktop application
   - Wait for it to fully start (whale icon should be steady)

2. **Docker service not running (Linux):**
   ```bash
   sudo systemctl start docker
   ```

3. **Permission issues (Linux):**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   # Log out and log back in
   ```

### Issue: "Permission denied while trying to connect to Docker daemon"

**Solution (Linux):**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker  # Or log out and back in
```

### Issue: "docker: Error response from daemon: ..."

**Solutions:**
- Restart Docker Desktop (Windows/Mac)
- Restart Docker service (Linux): `sudo systemctl restart docker`
- Check Docker logs for errors

## Verification Checklist

Use this checklist to verify Docker is fully functional:

- [ ] `docker --version` shows version number
- [ ] `docker info` displays system information
- [ ] `docker ps` runs without errors
- [ ] `docker run hello-world` successfully runs
- [ ] `docker-compose --version` or `docker compose version` works
- [ ] Docker Desktop is running (Windows/Mac) or Docker service is active (Linux)

## Quick Test Commands

Run these commands in sequence to verify Docker:

```bash
# 1. Check version
docker --version

# 2. Check daemon
docker info | head -5

# 3. List containers
docker ps -a

# 4. Test with hello-world
docker run --rm hello-world

# 5. Check Docker Compose
docker-compose --version || docker compose version
```

## Expected Results

### ✅ Docker Working Correctly

When Docker is working, you should see:
- Version information displayed
- No error messages
- Containers can be created and run
- Docker Compose available

### ❌ Docker Not Working

Common error messages:
- `docker: command not found` → Docker not installed
- `Cannot connect to Docker daemon` → Docker service not running
- `Permission denied` → User not in docker group (Linux)
- `Error response from daemon` → Docker daemon issues

## Next Steps

Once Docker is verified to be working:

1. **Build your application image:**
   ```bash
   docker build -t sign-to-text-speech .
   ```

2. **Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access your application:**
   Open browser to `http://localhost:5000`

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- [Docker Engine for Linux](https://docs.docker.com/engine/install/)
