"""
Simplified Token Embedding Visualization Script for BDH Architecture
Faster version with essential analyses only.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from bdh2 import BDH, BDHConfig

print("=" * 60)
print("BDH TOKEN EMBEDDING ANALYSIS (SIMPLIFIED)")
print("=" * 60)

# Configuration
MODEL_PATH = 'bdh_shakespeare_final.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = Path('results/plots/embedding_analysis')
save_dir.mkdir(parents=True, exist_ok=True)

# Load model
print(f"Loading model from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Handle compiled model
is_compiled = any('_orig_mod' in k for k in checkpoint.keys())
if is_compiled:
    print("Detected compiled model checkpoint")
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
else:
    state_dict = checkpoint

# Get config
vocab_size = state_dict['embed.weight'].shape[0]
n_embd = state_dict['embed.weight'].shape[1]
n_head = state_dict['encoder'].shape[0]

config = BDHConfig(vocab_size=vocab_size, n_layer=6, n_head=n_head, n_embd=n_embd)
model = BDH(config).to(DEVICE)
model.load_state_dict(state_dict, strict=False)

print(f"✓ Model loaded: vocab={vocab_size}, embd={n_embd}, heads={n_head}")

# Extract embeddings
print("\nExtracting embeddings...")
input_emb = model.embed.weight.detach().cpu().numpy()
output_emb = model.lm_head.detach().cpu().numpy().T

print(f"Input embeddings shape: {input_emb.shape}")
print(f"Output embeddings shape: {output_emb.shape}")

# Create simple vocabulary
vocab = [chr(i) if 32 <= i < 127 else f'<{i}>' for i in range(vocab_size)]

# 1. Embedding magnitude distribution
print("\n[1/5] Analyzing embedding magnitudes...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

input_norms = np.linalg.norm(input_emb, axis=1)
output_norms = np.linalg.norm(output_emb, axis=1)

axes[0].hist(input_norms, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].set_xlabel('L2 Norm', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title(f'Input Embedding Magnitudes\nMean: {input_norms.mean():.3f}, Std: {input_norms.std():.3f}', fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].hist(output_norms, bins=50, alpha=0.7, color='coral', edgecolor='black')
axes[1].set_xlabel('L2 Norm', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title(f'Output Embedding Magnitudes\nMean: {output_norms.mean():.3f}, Std: {output_norms.std():.3f}', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir / 'embedding_magnitudes.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved to embedding_magnitudes.png")
plt.close()

# 2. Similarity matrix (sample only for speed)
print("\n[2/5] Computing similarity matrix (first 50 tokens)...")
from sklearn.metrics.pairwise import cosine_similarity

n_sample = min(50, vocab_size)
sim_matrix = cosine_similarity(input_emb[:n_sample])

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(sim_matrix, cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=[vocab[i] if vocab[i].isprintable() else f'[{i}]' for i in range(n_sample)],
            yticklabels=[vocab[i] if vocab[i].isprintable() else f'[{i}]' for i in range(n_sample)],
            ax=ax, cbar_kws={'label': 'Cosine Similarity'})
ax.set_title(f'Token Embedding Similarity Matrix (First {n_sample} Tokens)', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(save_dir / 'embedding_similarity_sample.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved to embedding_similarity_sample.png")
plt.close()

# 3. Input vs Output comparison
print("\n[3/5] Comparing input and output embeddings...")
similarities = []
for i in range(vocab_size):
    sim = cosine_similarity(input_emb[i:i+1], output_emb[i:i+1])[0, 0]
    similarities.append(sim)

fig, ax = plt.subplots(figsize=(14, 6))
x_pos = np.arange(vocab_size)
bars = ax.bar(x_pos, similarities, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

for i, (bar, sim) in enumerate(zip(bars, similarities)):
    if sim > 0.8:
        bar.set_color('#2ECC71')
    elif sim < 0.5:
        bar.set_color('#E74C3C')

ax.set_xlabel('Token Index', fontsize=12)
ax.set_ylabel('Cosine Similarity', fontsize=12)
ax.set_title('Input vs Output Embedding Similarity (per token)', fontsize=14, pad=15)
ax.axhline(y=np.mean(similarities), color='blue', linestyle='--', alpha=0.5, 
           label=f'Mean: {np.mean(similarities):.3f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(save_dir / 'input_output_comparison.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved to input_output_comparison.png")
plt.close()

# 4. PCA visualization (faster than t-SNE)
print("\n[4/5] Performing PCA dimensionality reduction...")
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(input_emb)

fig, ax = plt.subplots(figsize=(14, 10))

# Categorize tokens
categories = []
colors_map = {
    'Letter': '#4ECDC4',
    'Digit': '#FFD93D',
    'Space': '#95E1D3',
    'Punct': '#A8E6CF',
    'Other': '#C7CEEA'
}

for i, char in enumerate(vocab):
    if len(char) == 1:
        if char.isalpha():
            categories.append('Letter')
        elif char.isdigit():
            categories.append('Digit')
        elif char in ' \n\t':
            categories.append('Space')
        elif char in '.,;:!?':
            categories.append('Punct')
        else:
            categories.append('Other')
    else:
        categories.append('Other')

colors = [colors_map[cat] for cat in categories]

ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

# Add labels for printable chars
for i, char in enumerate(vocab):
    if len(char) == 1 and char.isprintable() and char not in ' \n\t':
        ax.annotate(char, (coords[i, 0], coords[i, 1]), 
                   fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=cat, edgecolor='black') 
                  for cat, color in colors_map.items() if cat in categories]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.set_title('Token Input Embeddings (PCA)', fontsize=16, pad=20)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir / 'input_embeddings_pca.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved to input_embeddings_pca.png")
plt.close()

# 5. Most similar pairs
print("\n[5/5] Finding most similar token pairs...")
full_sim = cosine_similarity(input_emb)
triu_indices = np.triu_indices_from(full_sim, k=1)
similarities_flat = full_sim[triu_indices]
top_indices = np.argsort(similarities_flat)[-10:][::-1]

print("\nTop 10 Most Similar Token Pairs:")
print("-" * 50)
for idx in top_indices:
    i, j = triu_indices[0][idx], triu_indices[1][idx]
    char_i = vocab[i] if len(vocab[i]) == 1 and vocab[i].isprintable() else f'<{i}>'
    char_j = vocab[j] if len(vocab[j]) == 1 and vocab[j].isprintable() else f'<{j}>'
    print(f"  '{char_i}' ↔ '{char_j}': {similarities_flat[idx]:.4f}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print(f"All visualizations saved to: {save_dir}")
print("=" * 60)