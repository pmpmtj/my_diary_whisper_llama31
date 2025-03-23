"""
Configuration settings for the Llama 3.1 model for transcription processing
"""

# Model configuration
MODEL_CONFIG = {
    # Model ID from local directory
    "model_id": "C:/Users/pmpmt/models_in_mydiary/Llama-3.1-8B-Instruct",
    
    # Whether to use 4-bit quantization (reduces VRAM usage but slightly reduces quality)
    "use_4bit": False,
    
    # Whether to use 8-bit quantization
    "use_8bit": False,
    
    # Device map - "auto" will automatically place model on available devices
    "device_map": "auto",
}

# Generation configuration
GENERATION_CONFIG = {
    # Maximum number of new tokens to generate
    "max_new_tokens": 2048,
    
    # Temperature (higher = more creative, lower = more deterministic)
    "temperature": 0.3,
    
    # Top-p sampling (nucleus sampling)
    "top_p": 0.9,
    
    # Whether to use sampling at all (False = greedy decoding)
    "do_sample": True,
    
    # Repetition penalty (higher = less repetition)
    "repetition_penalty": 1.1,
} 