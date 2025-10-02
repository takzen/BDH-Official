# train.py
# Main training script for the BDH model.
# Uses advanced features like torch.compile and Automatic Mixed Precision.
# Includes checkpointing and a Windows-compatible compile backend.


import os
from contextlib import nullcontext
import torch
import numpy as np
import requests

import bdh # Import our model architecture from bdh.py

# --- Configuration Section ---
# These parameters can be adjusted for different experiments
BLOCK_SIZE = 512
BATCH_SIZE = 8       # Reduced to fit into 8GB VRAM
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
CHECKPOINT_FREQ = 1000 # Save a checkpoint every 1000 steps

# Compilation settings
USE_COMPILE = True  # Set to False to disable compilation
COMPILE_MODE = "default"  # Options: "default", "reduce-overhead", "max-autotune"

# File paths
MODEL_CHECKPOINT_PATH = "bdh_shakespeare_checkpoint.pth"
FINAL_MODEL_PATH = "bdh_shakespeare_final.pth"

# --- Device and Dtype Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = ("bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (torch.amp.autocast(device_type=device.type, dtype=ptdtype) if "cuda" in str(device) else nullcontext())
scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

# --- Performance Optimizations ---
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Using device: {device} with dtype: {dtype}")

# --- Data Loading Section ---
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

def fetch_data():
    """Downloads the Tiny Shakespeare dataset if it doesn't exist."""
    if not os.path.exists(input_file_path):
        print("Downloading Tiny Shakespeare dataset...")
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)
        print("Dataset downloaded.")

def get_batch(split):
    """Loads a batch of data using memory-mapping for efficiency."""
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    split_idx = int(0.9 * len(data))
    data = data[:split_idx] if split == "train" else data[split_idx:]
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64)) for i in ix])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

# --- Main Execution ---
if __name__ == "__main__":
    fetch_data()

    model_config = bdh.BDHConfig()
    model = bdh.BDH(model_config).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # --- Compile the model for a significant speed-up ---
    if USE_COMPILE:
        print(f"Compiling the model...")
        try:
            # Suppress dynamo errors to gracefully fall back if compilation fails
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            
            # On Windows, Triton is not available, so we use a compatible backend
            # 'aot_eager' works on all platforms without requiring Triton
            model = torch.compile(model, backend="aot_eager")
            print("Model compiled successfully with 'aot_eager' backend.")
        except Exception as e:
            print(f"Warning: torch.compile failed with error: {e}")
            print("Continuing without compilation...")
    else:
        print("Compilation disabled, running in eager mode.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"\nStarting training for {MAX_ITERS} iterations...")
    loss_acc = 0.0
    loss_steps = 0
    for step in range(MAX_ITERS):
        x, y = get_batch("train")
        with ctx:
            _, loss = model(x, y)
        loss_acc += loss.item()
        loss_steps += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Log progress
        if step > 0 and step % LOG_FREQ == 0:
            avg_loss = loss_acc / loss_steps
            print(f"Step: {step}/{MAX_ITERS} | loss: {avg_loss:.4f}")
            loss_acc = 0.0
            loss_steps = 0
            
        # --- Regular Checkpointing ---
        if step > 0 and step % CHECKPOINT_FREQ == 0:
            print(f"\n--- Saving checkpoint at step {step} ---")
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
            print(f"Model checkpoint saved to {MODEL_CHECKPOINT_PATH}")
            print("-----------------------------------------")

    print("\nTraining finished. Generating a sample...")
    model.eval()
    prompt = torch.tensor(bytearray("To be or not to be", "utf-8"), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        with ctx:
            ret = model.generate(prompt, max_new_tokens=200, top_k=5)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(errors="backslashreplace")
    print("-" * 50)
    print(ret_decoded)
    print("-" * 50)

    # --- Final Save ---
    print(f"\nSaving final model to {FINAL_MODEL_PATH}...")
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print("Final model saved successfully.")