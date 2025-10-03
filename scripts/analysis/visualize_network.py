# scripts/analysis/visualize_network.py
# Analyzes the internal network structure of a trained model.

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# --- Project Path Setup ---
# Ensures that imports from 'bdh', 'config', etc., work correctly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports from our project ---
from bdh import BDH, BDHConfig
from train import device, FINAL_MODEL_PATH # Use the path to our final model

def visualize(model_path: str):
    """
    Loads a trained model and generates a histogram of its internal
    neuron connection similarities.
    """
    print(f"--- Visualizing Network Structure ---")
    print(f"Loading model from: '{model_path}'")

    try:
        model_config = BDHConfig()
        model = BDH(model_config).to(device)
        
        # --- UNWRAPPING LOGIC FOR COMPILED/SAVED MODEL ---
        state_dict = torch.load(model_path, map_location=device)
        unwrapped_state_dict = {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}
        model.load_state_dict(unwrapped_state_dict)
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # We will analyze the 'encoder' matrix, but the logic can be applied to others.
    # Let's visualize the weights of the `decoder` this time for variety.
    # Shape: (nh * N, D)
    weights_to_analyze = model.decoder.detach().cpu().numpy()
    
    # Flatten all weights into a single list
    weights = weights_to_analyze.flatten()
    print(f"Analyzing {len(weights)} weights from the 'decoder' matrix...")

    # --- Create and save the plot ---
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.hist(weights, bins=200, log=True, color='darkgreen', alpha=0.75, label='Weight Frequency')
    ax.set_title('Weight Distribution of the Decoder Matrix (Log Scale)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Weight Value', fontsize=12)
    ax.set_ylabel('Count (Log Scale)', fontsize=12)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero')
    
    mean_weight = np.mean(weights)
    ax.axvline(mean_weight, color='blue', linestyle=':', linewidth=2, label=f'Mean ({mean_weight:.4f})')
    
    ax.legend()
    fig.tight_layout()

    # Save the plot to the results folder
    output_filename = f"network_weights_{os.path.basename(model_path).replace('.pth', '')}.png"
    output_path = os.path.join(project_root, 'results', 'plots', output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300)
    print(f"\n>>> Successfully saved plot to '{output_path}' <<<")
    plt.show()

if __name__ == '__main__':
    visualize(model_path=FINAL_MODEL_PATH)