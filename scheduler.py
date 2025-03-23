"""
Automated Audio Pipeline Scheduler

This script runs three operations sequentially based on the configured interval:
1. download-from-gdrive.py - Downloads audio files from Google Drive
2. local_whisper.py - Transcribes the downloaded audio files
3. process_transcription.py - Processes transcriptions with Llama 3.1 model

Author: [Your Name]
Date: [Current Date]
"""

import subprocess
import time
import logging
import os
import sys
import json
import traceback
from datetime import datetime, timedelta
# Import the FFmpeg path setup function
from ffmpeg_utils import setup_ffmpeg_path

# Configure FFmpeg path early
ffmpeg_path = setup_ffmpeg_path()

# Load configuration
def load_config():
    """Load configuration from config.json file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        # Return default configuration if config file is missing or invalid
        return {
            "scheduler": {
                "interval_seconds": 3600,
                "log_file": "pipeline_scheduler.log",
                "log_level": "INFO",
                "scripts": {
                    "download": "download-from-gdrive.py",
                    "transcribe": "local_whisper.py",
                    "process": "process_transcription.py"
                }
            }
        }

# Load config
config = load_config()
scheduler_config = config.get("scheduler", {})

# Configure logging
log_level_str = scheduler_config.get("log_level", "INFO")
log_level = getattr(logging, log_level_str)
log_file = scheduler_config.get("log_file", "pipeline_scheduler.log")

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Log FFmpeg status
if ffmpeg_path:
    logging.info(f"FFmpeg configured: {ffmpeg_path}")
else:
    logging.warning("FFmpeg not found! Audio processing may be affected.")

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the scripts to run (use absolute paths for reliability)
scripts = scheduler_config.get("scripts", {})
DOWNLOAD_SCRIPT_PATH = os.path.join(SCRIPT_DIR, scripts.get("download", "download-from-gdrive.py"))
WHISPER_SCRIPT_PATH = os.path.join(SCRIPT_DIR, scripts.get("transcribe", "local_whisper.py"))
PROCESS_SCRIPT_PATH = os.path.join(SCRIPT_DIR, scripts.get("process", "process_transcription.py"))

# Get the path to the Python executable that's running this script
# This ensures we use the same Python environment with all installed packages
PYTHON_EXECUTABLE = sys.executable

# Interval in seconds
INTERVAL = scheduler_config.get("interval_seconds", 3600)

def run_pipeline():
    """Run the complete pipeline: download files, transcribe audio, process with LLM"""
    logging.info("Starting pipeline execution")
    
    # Step 1: Download files from Google Drive
    logging.info("Step 1: Downloading files from Google Drive")
    try:
        download_process = subprocess.run(
            [PYTHON_EXECUTABLE, DOWNLOAD_SCRIPT_PATH],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',  # Explicitly set encoding to utf-8
            errors='replace'   # Replace characters that can't be decoded
        )
        logging.info(f"Download script output: {download_process.stdout}")
        if download_process.stderr:
            logging.warning(f"Download script errors: {download_process.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Download script failed with exit code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        # Continue to transcription anyway - there might be previously downloaded files
    
    # Step 2: Transcribe downloaded audio files
    logging.info("Step 2: Transcribing audio files")
    try:
        whisper_process = subprocess.run(
            [PYTHON_EXECUTABLE, WHISPER_SCRIPT_PATH],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',  # Explicitly set encoding to utf-8
            errors='replace'   # Replace characters that can't be decoded
        )
        logging.info(f"Transcription script output: {whisper_process.stdout}")
        if whisper_process.stderr:
            logging.warning(f"Transcription script errors: {whisper_process.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Transcription script failed with exit code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        return  # Stop here if transcription fails
    
    # Step 3: Process transcriptions with Llama 3.1
    logging.info("Step 3: Processing transcriptions with Llama 3.1")
    try:
        process_llm = subprocess.run(
            [PYTHON_EXECUTABLE, PROCESS_SCRIPT_PATH],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        logging.info(f"LLM processing script output: {process_llm.stdout}")
        if process_llm.stderr:
            logging.warning(f"LLM processing script errors: {process_llm.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"LLM processing script failed with exit code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
    
    logging.info("Pipeline execution completed")

def main():
    """Main function to run the scheduler"""
    logging.info(f"Audio Pipeline Scheduler started with {INTERVAL} second interval")
    
    try:
        while True:
            # Log start time for this cycle
            cycle_start = datetime.now()
            logging.info(f"Starting pipeline cycle at {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Run the pipeline
            run_pipeline()
            
            # Calculate time for next cycle
            elapsed = (datetime.now() - cycle_start).total_seconds()
            sleep_time = max(1, INTERVAL - elapsed)  # Ensure at least 1 second sleep
            
            # Calculate and display the next sync time
            next_sync_time = datetime.now() + timedelta(seconds=sleep_time)
            next_sync_time_formatted = next_sync_time.strftime("%H:%M")
            
            logging.info(f"Cycle completed in {elapsed:.2f} seconds. Next sync at {next_sync_time_formatted} (sleeping for {sleep_time:.2f} seconds).")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logging.info("Scheduler stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 