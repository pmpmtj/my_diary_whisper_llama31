"""
Process Transcriptions with Llama 3.1

This script:
1. Reads the transcription output from the local_whisper.py script
2. Processes it using the Llama 3.1 model
3. Adds the organized content to ongoing_entries.txt
4. Extracts to-do items to to_do.txt

Part of the audio diary pipeline.
"""

import os
import sys
import json
import torch
import logging
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_processor.log'),
        logging.StreamHandler()
    ]
)

def load_config():
    """Load the main configuration file"""
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return {}

def load_model_config():
    """Load the model configuration"""
    try:
        # Import model config from config.py
        from config import MODEL_CONFIG, GENERATION_CONFIG
        return MODEL_CONFIG, GENERATION_CONFIG
    except ImportError:
        logging.error("Could not import model config from config.py")
        # Default configuration if import fails
        return {
            "model_id": "C:/Users/pmpmt/models_in_mydiary/Llama-3.1-8B-Instruct",
            "use_4bit": False,
            "use_8bit": False,
            "device_map": "auto",
        }, {
            "max_new_tokens": 2048,
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1,
        }

def get_diary_organization_prompt(diary_entry, ongoing_entries):
    """Import the prompt template from prompts.py"""
    try:
        from prompts import get_diary_organization_prompt
        return get_diary_organization_prompt(diary_entry, ongoing_entries)
    except ImportError:
        logging.warning("Could not import prompt from prompts.py, using default prompt")
        # Default prompt if import fails
        prompt = f"""
        You are an intelligent diary organizer. Your task is to analyze a new diary entry and determine how it relates to previous entries. You should categorize and organize the content.

        # PREVIOUS DIARY ENTRIES:
        {ongoing_entries if ongoing_entries else "No previous entries exist yet."}

        # NEW DIARY ENTRY:
        {diary_entry}

        Please provide a detailed analysis with the following structure:

        ## ENTRY CATEGORIZATION
        - **Main Topics**: [Identify 1-2 main topics or themes in this entry]
        - **Emotional Tone**: [Analyze the emotional tone of the entry]
        - **Related Previous Entries**: [Identify any connections to previous entries]

        ## ORGANIZED ENTRY
        [Rewrite the entry with proper formatting while preserving all original content]

        ## TO-DO ITEMS
        [Extract any tasks, to-do items, or intentions mentioned in the entry. If none are found, write "No to-do items detected."]
        """
        return prompt

def extract_todo_items(analysis):
    """Extract to-do items from the model's analysis"""
    # Look for the TO-DO ITEMS section in the analysis
    todo_section_match = re.search(r'## TO-DO ITEMS\s+(.*?)(?=##|\Z)', analysis, re.DOTALL)
    
    if not todo_section_match:
        return []
        
    todo_section = todo_section_match.group(1).strip()
    
    # If no to-do items were found
    if "No to-do items detected" in todo_section:
        return []
        
    # Extract individual items (assuming they're in a list format)
    todo_items = []
    for line in todo_section.split('\n'):
        line = line.strip()
        if line.startswith('-') or line.startswith('*'):
            todo_items.append(line[1:].strip())
        elif re.match(r'^\d+\.', line):  # Numbered list
            todo_items.append(re.sub(r'^\d+\.', '', line).strip())
    
    return todo_items

def extract_organized_entry(analysis):
    """Extract the organized entry from the model's analysis"""
    # Look for the ORGANIZED ENTRY section in the analysis
    entry_section_match = re.search(r'## ORGANIZED ENTRY\s+(.*?)(?=##|\Z)', analysis, re.DOTALL)
    
    if not entry_section_match:
        return None
        
    organized_entry = entry_section_match.group(1).strip()
    return organized_entry

def read_transcription_file(config):
    """Read the transcription file specified in config.json"""
    try:
        output_file = config.get("output_file", "daily.txt")
        if not os.path.exists(output_file):
            logging.error(f"Transcription file {output_file} not found")
            return None
            
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if file has content
        if not content.strip():
            logging.warning(f"Transcription file {output_file} is empty")
            return None
            
        return content
    except Exception as e:
        logging.error(f"Error reading transcription file: {str(e)}")
        return None

def initialize_llm():
    """Initialize the Llama 3.1 model"""
    MODEL_CONFIG, GENERATION_CONFIG = load_model_config()
    
    # Set up quantization if needed
    quantization_config = None
    if MODEL_CONFIG.get("use_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif MODEL_CONFIG.get("use_8bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    try:
        # Load model and tokenizer
        logging.info("Loading Llama 3.1 model and tokenizer...")
        model_id = MODEL_CONFIG.get("model_id", "C:/Users/pmpmt/models_in_mydiary/Llama-3.1-8B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=MODEL_CONFIG.get("device_map", "auto"),
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
        
        # Create a text generation pipeline
        gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **GENERATION_CONFIG,
        )
        
        return gen_pipeline
    except Exception as e:
        logging.error(f"Error initializing LLM: {str(e)}")
        return None

def process_with_llm(transcription):
    """Process the transcription with Llama 3.1"""
    # Read ongoing entries
    ongoing_entries_path = "ongoing_entries.txt"
    ongoing_entries = ""
    
    if os.path.exists(ongoing_entries_path):
        try:
            with open(ongoing_entries_path, 'r', encoding='utf-8') as f:
                ongoing_entries = f.read()
        except Exception as e:
            logging.warning(f"Could not read ongoing entries file: {str(e)}")
    
    # Initialize the LLM
    gen_pipeline = initialize_llm()
    if gen_pipeline is None:
        logging.error("Failed to initialize LLM")
        return False
    
    # Get the organization prompt
    prompt = get_diary_organization_prompt(transcription, ongoing_entries)
    
    # Create a message list for the model
    messages = [
        {"role": "system", "content": "You are a helpful diary organization assistant that organizes entries and extracts to-do items."},
        {"role": "user", "content": prompt}
    ]
    
    # Generate analysis
    logging.info("Analyzing transcription with Llama 3.1...")
    result = gen_pipeline(messages)
    analysis = result[0]["generated_text"][-1]["content"]
    
    # Extract to-do items
    todo_items = extract_todo_items(analysis)
    if todo_items:
        logging.info(f"Found {len(todo_items)} to-do items")
        
        # Append to-do items to the to-do file
        try:
            todo_path = "to_do.txt"
            with open(todo_path, 'a', encoding='utf-8') as file:
                # Add timestamp to each to-do item
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                
                file.write(f"\n--- Added on {timestamp} ---\n")
                for item in todo_items:
                    file.write(f"- {item}\n")
            logging.info(f"To-do items saved to {todo_path}")
        except Exception as e:
            logging.error(f"Error saving to-do items: {str(e)}")
    else:
        logging.info("No to-do items found in the transcription")
    
    # Get the organized entry
    organized_entry = extract_organized_entry(analysis)
    if not organized_entry:
        logging.warning("Could not extract organized entry. Using original transcription.")
        organized_entry = transcription
    
    # Append the entry to the ongoing entries file
    try:
        with open(ongoing_entries_path, 'a', encoding='utf-8') as file:
            # Add a timestamp and separator if there's existing content
            if ongoing_entries:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                file.write(f"\n\n--- New Entry: {timestamp} ---\n\n")
            
            file.write(organized_entry)
        logging.info(f"Entry appended to {ongoing_entries_path}")
        return True
    except Exception as e:
        logging.error(f"Error appending entry: {str(e)}")
        return False

def main():
    """Main entry point"""
    logging.info("Starting LLM processing of transcription")
    
    # Load configuration
    config = load_config()
    
    # Read the transcription file
    transcription = read_transcription_file(config)
    if transcription is None:
        logging.error("No transcription to process")
        sys.exit(1)
    
    # Process the transcription with the LLM
    success = process_with_llm(transcription)
    
    if success:
        logging.info("Successfully processed transcription with Llama 3.1")
        
        # Clear the transcription file to prevent reprocessing
        try:
            output_file = config.get("output_file", "daily.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                pass  # Just open and close to clear the file
            logging.info(f"Cleared transcription file {output_file}")
        except Exception as e:
            logging.error(f"Error clearing transcription file: {str(e)}")
    else:
        logging.error("Failed to process transcription")
        sys.exit(1)
    
    logging.info("LLM processing complete")

if __name__ == "__main__":
    main() 