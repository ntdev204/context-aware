import argparse
import logging
import os
import subprocess
import time
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] auto-updater: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("auto_updater")

CORE_FILES = ["docker/", "requirements.txt", "Makefile", "env/"]

def has_core_changes() -> bool:
    """Check if the incoming update contains changes that require a Docker rebuild."""
    try:
        # Get list of changed files
        output = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "origin/main"]).decode("utf-8").strip()
        if not output:
            return False
            
        changed_files = output.split('\n')
        for f in changed_files:
            for core in CORE_FILES:
                if f.startswith(core) or f == core:
                    return True
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to check git diff: {e}")
        return True # Safe fallback: force build on error

def get_git_hashes():
    local = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    remote = subprocess.check_output(["git", "rev-parse", "origin/main"]).decode().strip()
    return local, remote

def run_update_sequence(target: str, needs_build: bool):
    logger.info("Executing update sequence...")
    
    # 1. Stop current containers
    logger.info(f"Stopping {target}...")
    subprocess.run(["make", f"{target}-down"])
    
    # 2. Pull new code
    logger.info("Running git pull...")
    subprocess.run(["git", "pull", "origin", "main"], check=True)
    
    # 3. Build if necessary
    if needs_build:
        logger.info(f"Core changes detected. Rebuilding {target}...")
        # target specific build commands
        build_cmd = f"{target}-build"
        if target == "jetson":
            build_cmd = "jetson-build-prod"
        subprocess.run(["make", build_cmd], check=True)
    else:
        logger.info("No core changes detected. Skipping build.")
        
    # 4. Restart
    logger.info(f"Restarting {target}...")
    subprocess.run(["make", f"{target}-up"], check=True)
    
    logger.info("Update sequence complete!")

def main():
    parser = argparse.ArgumentParser(description="Git Auto Updater Daemon")
    parser.add_argument("--target", choices=["jetson", "server"], required=True, help="Target environment to manage")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    args = parser.parse_args()

    logger.info(f"Auto Updater started for target '{args.target}' (interval: {args.interval}s)")

    while True:
        try:
            # Fetch latest from remote
            subprocess.run(["git", "fetch", "origin", "main"], check=True, capture_output=True)
            
            local, remote = get_git_hashes()
            
            if local != remote:
                logger.info(f"Update detected! Remote has moved from {local[:7]} to {remote[:7]}")
                needs_build = has_core_changes()
                
                run_update_sequence(args.target, needs_build)
            
        except Exception as e:
            logger.error(f"Error in update loop: {e}")
            
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
