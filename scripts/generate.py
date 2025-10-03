# scripts/generate.py
# FINAL ROBUST VERSION: Generates text and saves it to a structured results folder.

import torch
import sys
import os
import datetime
from contextlib import nullcontext

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bdh import BDH, BDHConfig
from train import device, ctx, FINAL_MODEL_PATH

def generate(prompt: str, model_path: str, max_new_tokens: int = 500, top_k: int = 10, temperature: float = 0.8):
    """
    Loads a trained model, generates text, prints it, and saves it to a file.
    """
    print(f"--- Generating Text ---")
    print(f"Using device: {device}")
    
    model_path_full = os.path.join(project_root, model_path)
    print(f"Loading model from: '{model_path_full}'")

    try:
        model_config = BDHConfig()
        model = BDH(model_config).to(device)
        
        # --- UNWRAPPING LOGIC FOR COMPILED MODEL ---
        state_dict = torch.load(model_path_full, map_location=device)
        unwrapped_state_dict = {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}
        model.load_state_dict(unwrapped_state_dict)
        # --- END OF LOGIC ---
        
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path_full}'.")
        return

    context = torch.tensor(bytearray(prompt, 'utf-8'), dtype=torch.long, device=device).unsqueeze(0)

    print(f"\nPrompt: '{prompt}'")
    print(f"Generating {max_new_tokens} new tokens (top_k={top_k}, temp={temperature})...")
    
    with torch.no_grad(), ctx:
        generated_bytes = model.generate(
            context, 
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature
        )[0].tolist()

    try:
        generated_text = bytearray(generated_bytes).decode('utf-8', errors='replace')
        
        print("\n" + "="*80)
        print("GENERATED TEXT:")
        print("="*80)
        print(generated_text)
        print("="*80)

        # --- NEW: SAVING THE OUTPUT ---
        # Create a unique filename with a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(model_path).replace('.pth', '')
        output_filename = f"{model_name}_{timestamp}.txt"
        output_path = os.path.join(project_root, 'results', 'generated_texts', output_filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write the generated text to the file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write("="*80 + "\n")
            f.write(generated_text)
            
        print(f"\nSuccessfully saved generated text to '{output_path}'")
        # --- END OF NEW LOGIC ---

    except Exception as e:
        print(f"An error occurred during text decoding or saving: {e}")

if __name__ == '__main__':
    model_to_use = FINAL_MODEL_PATH
    start_prompt = "Shall I compare thee to a summer's day?"
    generate(prompt=start_prompt, model_path=model_to_use)