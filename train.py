# train.py
# Main training script for the BDH model.
# Uses advanced features like torch.compile and Automatic Mixed Precision.
# Copyright 2025 Pathway Technology, Inc.

import os
from contextlib import nullcontext

import bdh # Import our model architecture from bdh.py
import numpy as np
import requests
import torch

# --- Configuration Section ---

# We can define parameters directly here for simplicity
BLOCK_SIZE = 512
BATCH_SIZE = 8
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100

# --- Device and Dtype Setup ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)

# A context manager for automatic mixed precision
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in str(device) # More robust check
    else nullcontext()
)

# A scaler for float16 training to prevent underflow
scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

# --- Performance Optimizations ---
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True # Allow TF32 on cuDNN

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
    
    # Split data into training (90%) and validation (10%) sets
    split_idx = int(0.9 * len(data))
    if split == "train":
        data = data[:split_idx]
    else:
        data = data[split_idx:]
        
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + BLOCK_SIZE]).astype(np.int64)) for i in ix]
    )
    
    # Move data to GPU asynchronously for performance
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

# --- Main Execution ---

if __name__ == "__main__":
    fetch_data()

    # 1. Initialize the model
    model_config = bdh.BDHConfig()
    model = bdh.BDH(model_config).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # 2. Compile the model for a significant speed-up (requires PyTorch 2.0+)
    print("Compiling the model...")
    # model = torch.compile(model)
    print("Model compiled.")

    # 3. Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 4. Training loop
    print(f"\nStarting training for {MAX_ITERS} iterations...")
    loss_acc = 0.0
    loss_steps = 0
    
    for step in range(MAX_ITERS):
        # Get a batch of data
        x, y = get_batch("train")
        
        # Forward pass with automatic mixed precision
        with ctx:
            logits, loss = model(x, y)
        
        # Accumulate loss for logging
        loss_acc += loss.item()
        loss_steps += 1
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True) # Use set_to_none for better performance
        
        # Log progress
        if step % LOG_FREQ == 0 and step > 0:
            avg_loss = loss_acc / loss_steps
            print(f"Step: {step}/{MAX_ITERS} | loss: {avg_loss:.4f}")
            loss_acc = 0.0
            loss_steps = 0

    # 5. Generate a sample after training
    print("\nTraining finished. Generating a sample...")
    model.eval()
    prompt = torch.tensor(
        bytearray("To be or not to be", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    
    with torch.no_grad():
        with ctx:
            ret = model.generate(prompt, max_new_tokens=200, top_k=5)
    
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print("-" * 50)
    print(ret_decoded)
    print("-" * 50)