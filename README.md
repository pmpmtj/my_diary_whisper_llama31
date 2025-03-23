# Audio Diary with Whisper Transcription and Llama 3.1 Summarization

A complete audio diary system that:
1. Downloads audio recordings from Google Drive
2. Transcribes them using OpenAI's Whisper (locally)
3. Processes the transcriptions with Llama 3.1 to organize and extract insights
4. Maintains an ongoing record of diary entries and to-do items

## Features

- **Google Drive Integration**: Automatically downloads audio files from specified folders
- **Local Whisper Transcription**: Converts audio to text without sending data to external APIs
- **LLM-Powered Analysis**: Uses Llama 3.1 to organize transcriptions and extract to-do items
- **Scheduled Operation**: Runs automatically at configurable intervals with next sync time display
- **Flexible Configuration**: Easy to customize via config files
- **Date-Based File Management**: Creates separate diary files for each day with YYMMDD date prefixes

## Quick Start

### Prerequisites

- Python 3.7 or higher
- PyTorch with CUDA support (recommended for GPU acceleration)
- Google Drive API credentials
- FFmpeg installed or available in the system path
- Local copies of Whisper and Llama 3.1 models

### Installation

1. Clone this repository
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up Google Drive API credentials:
   - Create a project in Google Cloud Console
   - Enable the Google Drive API
   - Create OAuth credentials and download as `credentials.json`
   - Place `credentials.json` in the project directory

4. Set up the local models:
   - Place your Whisper model in the directory specified in `config.json`
   - Ensure the Llama 3.1 model path is correctly set in `config.py`

5. Configure the system:
   - Edit `config.json` to specify Google Drive folders and settings
   - Edit `config.py` to customize the Llama 3.1 model configuration

6. Ensure FFmpeg is installed:
   - The system will try to find FFmpeg in your system PATH
   - Alternatively, place `ffmpeg.exe` in the project directory (Windows)
   - The scheduler will automatically configure FFmpeg at startup

### Running the Pipeline

Start the automated scheduler to run the pipeline at regular intervals:
```
python scheduler.py
```

The scheduler will:
- Download new audio files from Google Drive
- Transcribe them using Whisper
- Process the transcriptions with Llama 3.1
- Display the next scheduled sync time
- Repeat at the interval specified in config.json

### Individual Components

Run each step separately if needed:
```
python download-from-gdrive.py  # Download audio files
python local_whisper.py         # Transcribe audio files
python process_transcription.py # Process with Llama 3.1
```

### Managing Diary Files

The system automatically creates date-prefixed diary files (e.g., `240323_ongoing_entries.txt`) for each day.

To manually set the date (useful after system downtime):
```
python scheduler.py --set-date YYMMDD
```

For example, to set the date to March 23, 2024:
```
python scheduler.py --set-date 240323
```

If no date is specified, it will use today's date:
```
python scheduler.py --set-date
```

## How It Works

### 1. Audio Download

The `download-from-gdrive.py` script:
- Authenticates with Google Drive using OAuth
- Searches for audio files in configured folders
- Downloads files to the local `downloads` directory

### 2. Transcription

The `local_whisper.py` script:
- Loads a local Whisper model
- Transcribes all audio files in the `downloads` directory
- Saves transcriptions to a single file
- Moves processed audio files to the `processed_audio` directory

### 3. LLM Processing

The `process_transcription.py` script:
- Loads the Llama 3.1 model
- Reads the transcription file
- Processes the content to organize and analyze it
- Extracts to-do items and saves them to `to_do.txt`
- Adds the organized entry to the appropriate date-prefixed diary file

### 4. Scheduling

The `scheduler.py` script:
- Runs the complete pipeline at regular intervals
- Logs all operations for monitoring and debugging
- Displays the next scheduled run time
- Manages the sequence of operations
- Handles day changes and creates new diary files as needed

### 5. Date Management

The system:
- Tracks the current date in `config.json`
- Creates a new date-prefixed file each day
- Detects date changes even after system downtime
- Provides a utility for manually setting dates when needed

## Configuration

### Main Configuration (config.json)

```json
{
  "downloads_directory": "./downloads",
  "output_file": "transcription.txt",
  "processed_directory": "./processed_audio",
  "model": {
    "folder": "path/to/whisper/model",
    "name": "my_model.pt"
  },
  "diary_manager": {
    "current_date": "240323",
    "entries_file_format": "{date}_ongoing_entries.txt",
    "legacy_file": "ongoing_entries.txt",
    "auto_update_date": true
  },
  "scheduler": {
    "interval_seconds": 3600,
    "scripts": {
      "download": "download-from-gdrive.py",
      "transcribe": "local_whisper.py",
      "process": "process_transcription.py"
    }
  }
}
```

### LLM Configuration (config.py)

```python
MODEL_CONFIG = {
    "model_id": "path/to/llama/model",
    "use_4bit": False,
    "use_8bit": False,
    "device_map": "auto",
}

GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "temperature": 0.3,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}
```

## Output Files

- **transcription.txt**: Raw transcriptions from audio files
- **YYMMDD_ongoing_entries.txt**: Date-prefixed organized diary entries for each day
- **to_do.txt**: Extracted to-do items and action points

## Troubleshooting

- **Google Drive Authentication**: If authorization fails, delete `token.pickle` and restart
- **Model Loading Issues**: Check paths in configuration files and adjust memory settings
- **Transcription Quality**: Adjust Whisper settings in `config.json` for better results
- **LLM Performance**: Modify quantization settings in `config.py` based on your hardware
- **FFmpeg Issues**: Make sure FFmpeg is installed and in your PATH, or place ffmpeg.exe in the project directory
- **Date Management**: If the system was down for multiple days, use the `--set-date` utility to set the correct date

## System Requirements

- **Minimum**: 8GB RAM, modern CPU
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **Disk Space**: ~10GB for models and audio files
- **FFmpeg**: Required for audio processing (automatically detected or can be placed in project directory)

## License

[Insert license information]

---

For issues, suggestions, or contributions, please open an issue in the repository. 