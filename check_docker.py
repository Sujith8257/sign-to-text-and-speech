"""
Docker Installation Checker
Checks if Docker is installed and working correctly
"""

import subprocess
import sys
import platform

def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=10
        )
        stdout = result.stdout.strip().encode('ascii', errors='replace').decode('ascii')
        stderr = result.stderr.strip().encode('ascii', errors='replace').decode('ascii')
        return result.returncode == 0, stdout, stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_docker_installed():
    """Check if Docker is installed"""
    print("=" * 60)
    print("Docker Installation Checker")
    print("=" * 60)
    print()
    
    # Check Docker version
    print("1. Checking Docker installation...")
    success, output, error = run_command("docker --version")
    if success:
        print(f"   [OK] Docker is installed: {output}")
    else:
        print(f"   [ERROR] Docker is NOT installed")
        print(f"     Error: {error}")
        print()
        print("   Please install Docker from: https://www.docker.com/get-started")
        return False
    print()
    
    # Check Docker daemon
    print("2. Checking Docker daemon...")
    success, output, error = run_command("docker info")
    if success:
        print("   [OK] Docker daemon is running")
        # Extract useful info
        lines = output.split('\n')
        for line in lines[:10]:  # Show first 10 lines
            if any(keyword in line.lower() for keyword in ['containers', 'images', 'version', 'server']):
                print(f"     {line.strip()}")
    else:
        print("   [ERROR] Docker daemon is NOT running")
        print(f"     Error: {error}")
        print()
        print("   Please start Docker Desktop (Windows/Mac) or Docker service (Linux)")
        return False
    print()
    
    # Check Docker Compose
    print("3. Checking Docker Compose...")
    success, output, error = run_command("docker-compose --version")
    if not success:
        success, output, error = run_command("docker compose version")
    
    if success:
        print(f"   [OK] Docker Compose is installed: {output}")
    else:
        print("   [WARNING] Docker Compose not found (optional, but recommended)")
        print(f"     Error: {error}")
    print()
    
    # Test Docker with hello-world
    print("4. Testing Docker with hello-world container...")
    success, output, error = run_command("docker run --rm hello-world")
    if success:
        print("   [OK] Docker is working correctly!")
        if "Hello from Docker!" in output:
            print("     Container ran successfully")
    else:
        print("   [ERROR] Docker test failed")
        print(f"     Error: {error}")
        return False
    print()
    
    # Check system info
    print("5. System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print()
    
    print("=" * 60)
    print("[SUCCESS] All checks passed! Docker is ready to use.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        check_docker_installed()
    except KeyboardInterrupt:
        print("\n\nCheck interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during check: {e}")
        sys.exit(1)
