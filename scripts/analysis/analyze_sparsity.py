# scripts/analysis/analyze_sparsity.py
# Version with improved, readable token labels

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports from our project ---
from bdh2 import BDH, BDHConfig
from train import device, ctx, FINAL_MODEL_PATH

# --- Global variable for the hook ---
captured_activations = None

def activation_hook(module, input, output):
    """Hook to capture the output of our hook point (which is 'x_sparse')."""
    global captured_activations
    captured_activations = output.detach().cpu()

def analyze_sparsity(model_path: str):
    """Loads a trained model and analyzes its activation sparsity."""
    print(f"--- Sparsity Analysis ---")
    print(f"Loading model from '{model_path}' into the instrumented BDH2 architecture...")
    model_path_full = os.path.join(project_root, model_path)
    try:
        model_config = BDHConfig()
        model = BDH(model_config).to(device)
        state_dict = torch.load(model_path_full, map_location=device)
        unwrapped_state_dict = {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}
        model.load_state_dict(unwrapped_state_dict)
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path_full}'.")
        return

    hook_handle = model.x_sparse_hook_point.register_forward_hook(activation_hook)
    print("Hook registered on 'x_sparse_hook_point'.")

    repetitive_part = "The same old story, again and again. "
    surprise_part = "BUT SUDDENLY, EVERYTHING CHANGED! "
    test_text = repetitive_part * 2 + surprise_part + repetitive_part * 2
    print(f"\nAnalyzing text: '{test_text}'")
    
    sparsity_per_token = []
    with torch.no_grad(), ctx:
        context = torch.tensor(bytearray(test_text, 'utf-8'), dtype=torch.long, device=device).unsqueeze(0)
        model(context)
        B, n_head, T, N_per_head = captured_activations.shape
        all_neurons = captured_activations.transpose(1, 2).reshape(B, T, -1).squeeze(0)
        for t in range(T):
            token_activations = all_neurons[t]
            active_neurons_fraction = torch.count_nonzero(token_activations) / token_activations.numel()
            sparsity_per_token.append(active_neurons_fraction.item())
    hook_handle.remove()

    print("\nGenerating improved sparsity plot...")
    tokens_str = list(test_text)
    display_tokens = tokens_str  
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(28, 12))
    
    ax.plot(sparsity_per_token, marker='o', markersize=4, linestyle='-', 
            color='teal', linewidth=2, label='Fraction of Active Neurons')
    
    ax.set_title('Neuron Activation Sparsity Responds to "Surprise" Events', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Input Token (Character)', fontsize=16, labelpad=25)
    ax.set_ylabel('Sparsity (Fraction of Active Neurons)', fontsize=16, labelpad=15)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    ax.set_xticks(range(len(display_tokens)))
    ax.set_xticklabels(display_tokens, 
                       rotation=0, 
                       fontsize=10,
                       fontfamily='monospace',
                       ha='center')
    
    start_surprise = len(repetitive_part * 2)
    end_surprise = start_surprise + len(surprise_part)
    ax.axvspan(start_surprise - 0.5, end_surprise - 0.5, 
               color='orange', alpha=0.3, label='"Surprise" Event')
    
    for i in range(0, len(tokens_str), 10):
        ax.axvline(x=i, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    ax.legend(fontsize=11, loc='upper right')
    
    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.06, right=0.98)
    
    output_path = os.path.join(project_root, 'results', 'plots', 'sparsity_analysis_final.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n>>> Improved plot saved to '{output_path}' <<<")
    plt.show()

if __name__ == '__main__':
    analyze_sparsity(model_path=FINAL_MODEL_PATH)