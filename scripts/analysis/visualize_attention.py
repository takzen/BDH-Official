"""
Attention Pattern Visualization Script for BDH Architecture

This script loads a trained BDH model and visualizes the attention patterns.
Note: BDH uses a custom attention mechanism with RoPE, not standard multi-head attention.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from bdh2.py (adjust if your file has different name)
from bdh2 import BDH, BDHConfig


def get_attention_scores(model, text, device='cuda'):
    """
    Extract attention scores from BDH model by monkey-patching the attention forward.
    
    Args:
        model: Trained BDH model
        text: Input text string
        device: Device to run on
    
    Returns:
        attention_scores: List of attention score matrices, one per layer
        tokens: List of token strings for visualization
    """
    print(f"Processing text of length {len(text)}...")
    model.eval()
    
    # Encode text
    print("Encoding text...")
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s]
    
    # Prepare input
    context = torch.tensor([encode(text)], dtype=torch.long, device=device)
    print(f"Context shape: {context.shape}")
    
    attention_scores_list = []
    
    # Save original forward method
    original_forward = model.attn.forward
    
    # Create wrapper that captures scores
    def attention_forward_wrapper(Q, K, V):
        print(f"  Attention called! Layer {len(attention_scores_list) + 1}")
        print(f"    Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
        
        # Call original forward to get output
        output = original_forward(Q, K, V)
        
        # Now calculate scores separately for visualization
        _, _, T, _ = Q.size()
        r_phases = (torch.arange(0, T, device=model.attn.freqs.device, dtype=model.attn.freqs.dtype).view(1, 1, -1, 1)) * model.attn.freqs
        QR = model.attn.rope(r_phases, Q)
        KR = QR
        scores = (QR @ KR.mT).tril(diagonal=-1)
        
        # Normalize to [0, 1] for visualization
        scores_normalized = torch.softmax(scores, dim=-1)
        attention_scores_list.append(scores_normalized.detach().cpu())
        print(f"    ✓ Captured attention matrix of shape {scores_normalized.shape}")
        
        return output
    
    # Replace forward method temporarily
    print("Patching attention forward method...")
    model.attn.forward = attention_forward_wrapper
    
    # Forward pass - attention is called n_layer times
    print("Running forward pass...")
    try:
        with torch.no_grad():
            _ = model(context)
        print(f"Forward pass complete! Captured {len(attention_scores_list)} attention matrices")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Restore original forward
        model.attn.forward = original_forward
    
    return attention_scores_list, list(text)


def visualize_attention_layer(attention_scores, tokens, layer_idx, save_dir):
    """
    Visualize attention patterns for a single layer.
    
    Args:
        attention_scores: Attention matrix (B, nh, T, T)
        tokens: List of token strings
        layer_idx: Layer index for labeling
        save_dir: Directory to save plots
    """
    # Average across all attention heads
    att = attention_scores[0].mean(dim=0).numpy()  # (T, T)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Full heatmap
    sns.heatmap(att, xticklabels=tokens, yticklabels=tokens, 
                cmap='viridis', ax=axes[0], cbar_kws={'label': 'Attention Score'})
    axes[0].set_title(f'Layer {layer_idx} - Full Attention Pattern (BDH)', fontsize=14, pad=10)
    axes[0].set_xlabel('Key Position', fontsize=12)
    axes[0].set_ylabel('Query Position', fontsize=12)
    
    # Zoomed view - last 20 tokens if sequence is long
    if len(tokens) > 20:
        att_zoom = att[-20:, -20:]
        tokens_zoom = tokens[-20:]
    else:
        att_zoom = att
        tokens_zoom = tokens
    
    sns.heatmap(att_zoom, xticklabels=tokens_zoom, yticklabels=tokens_zoom,
                cmap='viridis', ax=axes[1], cbar_kws={'label': 'Attention Score'})
    axes[1].set_title(f'Layer {layer_idx} - Zoomed (Last 20 Tokens)', fontsize=14, pad=10)
    axes[1].set_xlabel('Key Position', fontsize=12)
    axes[1].set_ylabel('Query Position', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f'attention_layer_{layer_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved attention pattern for layer {layer_idx} to {save_path}")
    plt.close()


def visualize_attention_heads(attention_scores, tokens, layer_idx, save_dir):
    """
    Visualize individual attention heads for a layer.
    
    Args:
        attention_scores: Attention matrix (B, nh, T, T)
        tokens: List of token strings
        layer_idx: Layer index
        save_dir: Directory to save plots
    """
    n_heads = attention_scores.shape[1]
    att = attention_scores[0].numpy()  # (nh, T, T)
    
    # Create grid of subplots
    n_cols = 4
    n_rows = (n_heads + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_heads > 1 else [axes]
    
    for head_idx in range(n_heads):
        ax = axes[head_idx]
        
        # Use last 15 tokens for clarity
        if len(tokens) > 15:
            att_head = att[head_idx, -15:, -15:]
            tokens_vis = tokens[-15:]
        else:
            att_head = att[head_idx]
            tokens_vis = tokens
        
        sns.heatmap(att_head, xticklabels=tokens_vis, yticklabels=tokens_vis,
                    cmap='viridis', ax=ax, cbar=True)
        ax.set_title(f'Head {head_idx}', fontsize=12)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
    
    # Hide unused subplots
    for idx in range(n_heads, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Layer {layer_idx} - Individual Attention Heads (BDH)', fontsize=16, y=1.00)
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f'attention_heads_layer_{layer_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved attention heads for layer {layer_idx} to {save_path}")
    plt.close()


def analyze_attention_statistics(attention_scores_list, tokens, save_dir):
    """
    Analyze and visualize statistics about attention patterns.
    
    Args:
        attention_scores_list: List of attention matrices
        tokens: List of tokens
        save_dir: Directory to save plots
    """
    n_layers = len(attention_scores_list)
    
    # Calculate metrics per layer
    avg_attention_distance = []
    max_attention_distance = []
    attention_entropy = []
    
    for att in attention_scores_list:
        att_mean = att[0].mean(dim=0).numpy()  # Average across heads
        T = att_mean.shape[0]
        
        # Average attention distance (how far back does the model look?)
        distances = []
        for i in range(T):
            for j in range(i + 1):  # Only look at causal attention
                if att_mean[i, j] > 0.01:  # Threshold for significant attention
                    distances.append(i - j)
        
        avg_attention_distance.append(np.mean(distances) if distances else 0)
        max_attention_distance.append(np.max(distances) if distances else 0)
        
        # Entropy of attention distribution (how focused vs distributed?)
        entropy = []
        for i in range(T):
            p = att_mean[i, :i+1]  # Causal mask
            p = p / (p.sum() + 1e-10)
            ent = -np.sum(p * np.log(p + 1e-10))
            entropy.append(ent)
        attention_entropy.append(np.mean(entropy))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    layers = list(range(n_layers))
    
    # Average attention distance
    axes[0].plot(layers, avg_attention_distance, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('Average Distance (tokens)', fontsize=12)
    axes[0].set_title('Average Attention Distance (BDH)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Max attention distance
    axes[1].plot(layers, max_attention_distance, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('Max Distance (tokens)', fontsize=12)
    axes[1].set_title('Maximum Attention Distance', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Attention entropy
    axes[2].plot(layers, attention_entropy, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Layer', fontsize=12)
    axes[2].set_ylabel('Entropy (nats)', fontsize=12)
    axes[2].set_title('Attention Entropy (Focus vs Distribution)', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / 'attention_statistics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved attention statistics to {save_path}")
    plt.close()


def visualize_rope_patterns(model, save_dir):
    """
    Visualize RoPE (Rotary Position Encoding) patterns used in BDH.
    
    Args:
        model: BDH model
        save_dir: Directory to save plots
    """
    # Get frequency values
    freqs = model.attn.freqs[0, 0, 0, :].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Frequency spectrum
    axes[0].plot(freqs, linewidth=2)
    axes[0].set_xlabel('Dimension Index', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('RoPE Frequency Spectrum', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Log scale
    axes[1].semilogy(freqs, linewidth=2)
    axes[1].set_xlabel('Dimension Index', fontsize=12)
    axes[1].set_ylabel('Frequency (log scale)', fontsize=12)
    axes[1].set_title('RoPE Frequency Spectrum (Log Scale)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = save_dir / 'rope_frequency_spectrum.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved RoPE frequency spectrum to {save_path}")
    plt.close()


def main():
    # Configuration
    MODEL_PATH = 'bdh_shakespeare_final.pth'
    SAMPLE_TEXT = """ROMEO:
O, she doth teach the torches to burn bright!
It seems she hangs upon the cheek of night"""
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory
    save_dir = Path('results/plots/attention_analysis')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BDH ATTENTION PATTERN VISUALIZATION")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Sample text length: {len(SAMPLE_TEXT)} characters")
    print()
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Debug: print checkpoint keys
    first_key = list(checkpoint.keys())[0] if checkpoint else ''
    print(f"First checkpoint key: {first_key}")
    
    # Check if this is a compiled model (has _orig_mod prefix)
    is_compiled = any('_orig_mod' in k for k in checkpoint.keys())
    
    if is_compiled:
        print("Detected compiled model checkpoint (_orig_mod prefix)")
        # Remove the _orig_mod prefix
        state_dict = {}
        for key, value in checkpoint.items():
            new_key = key.replace('_orig_mod.', '')
            state_dict[new_key] = value
    else:
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Try to extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'vocab_size' in checkpoint:
        config = BDHConfig(
            vocab_size=checkpoint['vocab_size'],
            n_layer=checkpoint['n_layer'],
            n_head=checkpoint['n_head'],
            n_embd=checkpoint['n_embd']
        )
    else:
        # Infer from state dict
        # Find embed key
        embed_key = 'embed.weight' if 'embed.weight' in state_dict else None
        if not embed_key:
            for key in state_dict.keys():
                if 'embed' in key.lower() and 'weight' in key:
                    embed_key = key
                    break
        
        if embed_key:
            vocab_size = state_dict[embed_key].shape[0]
            n_embd = state_dict[embed_key].shape[1]
            print(f"Found embed: {embed_key}, vocab_size={vocab_size}, n_embd={n_embd}")
        else:
            vocab_size = 256
            n_embd = 256
        
        # Find encoder to get n_head
        encoder_key = 'encoder' if 'encoder' in state_dict else None
        n_head = state_dict[encoder_key].shape[0] if encoder_key else 4
        n_layer = 6  # Default for BDH
        
        config = BDHConfig(
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd
        )
        print(f"Inferred config: vocab={vocab_size}, n_embd={n_embd}, n_layer={n_layer}, n_head={n_head}")
    
    model = BDH(config).to(DEVICE)
    model.load_state_dict(state_dict, strict=False)
    
    print("Model loaded successfully!")
    print(f"Architecture: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embd dim")
    print()
    
    # Visualize RoPE patterns
    print("Visualizing RoPE frequency patterns...")
    try:
        visualize_rope_patterns(model, save_dir)
    except Exception as e:
        print(f"Warning: Could not visualize RoPE patterns: {e}")
    print()
    
    # Get attention patterns
    print("Extracting attention patterns...")
    try:
        attention_scores_list, tokens = get_attention_scores(model, SAMPLE_TEXT, DEVICE)
        print(f"✓ Extracted attention from {len(attention_scores_list)} layers")
        
        if len(attention_scores_list) == 0:
            print("WARNING: No attention matrices captured! Skipping visualizations.")
            print("This might be because the hook didn't capture data properly.")
            return
        
        print(f"First attention matrix shape: {attention_scores_list[0].shape}")
        
    except Exception as e:
        print(f"✗ Error extracting attention: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # Visualize each layer
    print("Generating visualizations...")
    for layer_idx, att in enumerate(attention_scores_list):
        print(f"  Processing layer {layer_idx}...")
        try:
            visualize_attention_layer(att, tokens, layer_idx, save_dir)
            
            # Also visualize individual heads for first and last layer
            if layer_idx == 0 or layer_idx == len(attention_scores_list) - 1:
                visualize_attention_heads(att, tokens, layer_idx, save_dir)
        except Exception as e:
            print(f"  Warning: Could not visualize layer {layer_idx}: {e}")
    
    # Analyze statistics
    print("Generating attention statistics...")
    try:
        analyze_attention_statistics(attention_scores_list, tokens, save_dir)
    except Exception as e:
        print(f"Warning: Could not generate statistics: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"All visualizations saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()