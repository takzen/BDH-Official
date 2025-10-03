# scripts/analysis/analyze_sparsity.py
# Analyzes activation sparsity using the instrumented 'bdh2.py' model definition.

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- IMPORTANT: We import the instrumented model from bdh2 ---
from bdh2 import BDH, BDHConfig
from train import device, ctx, FINAL_MODEL_PATH

captured_activations = None

def activation_hook(module, input, output):
    global captured_activations
    captured_activations = output.detach().cpu()

def analyze_sparsity(model_path: str):
    print(f"Loading model from '{model_path}' into the instrumented BDH architecture...")
    try:
        # We build the model using the BDH class from bdh2.py
        model = BDH(BDHConfig()).to(device)
        # We load the weights saved from the original bdh.py model.
        # This works because the only difference is a non-trainable Identity layer.
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        return

    # Register the hook on the hook point that exists only in our bdh2.py version
    hook_handle = model.x_sparse_hook_point.register_forward_hook(activation_hook)
    print("Hook registered successfully on 'x_sparse_hook_point'.")

    repetitive_part = "The same old thing. "
    surprise_part = "SUDDENLY, A WILD PLOT TWIST! "
    test_text = repetitive_part * 3 + surprise_part + repetitive_part * 2
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

    print("\nGenerating sparsity plot...")
    tokens_str = list(test_text)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(sparsity_per_token, marker='.', linestyle='-', color='teal', label='Fraction of Active Neurons')
    ax.set_title('Neuron Activation Sparsity Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Input Token (Character)', fontsize=12)
    ax.set_ylabel('Sparsity (Fraction of Active Neurons)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(range(len(tokens_str)))
    ax.set_xticklabels(tokens_str, rotation=90, fontdict={'fontsize': 8, 'fontfamily': 'monospace'})
    start_surprise = len(repetitive_part * 3)
    end_surprise = start_surprise + len(surprise_part)
    ax.axvspan(start_surprise - 1, end_surprise, color='gold', alpha=0.3, label='"Surprise" Event')
    ax.legend()
    fig.tight_layout()
    
    output_path = "results/plots/sparsity_analysis.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to '{output_path}'")
    plt.show()

if __name__ == '__main__':
    analyze_sparsity(model_path=FINAL_MODEL_PATH)