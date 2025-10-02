# BDH-Official: An Open-Source Implementation of the "Dragon Hatchling" AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains an open-source project built upon the **official reference implementation** of the **"Dragon Hatchling" (BDH)** AI architecture, released by Pathway Technology, Inc. on their GitHub. This framework provides a professional environment to train, analyze, and explore the fascinating phenomenon of **emergence** in neural networks, as described in their conceptual paper.

## Core Idea: Emergence

The project is built on the hypothesis that complex, functional structures can **emerge** from simple, local rules during a neural network's training. Instead of being explicitly programmed, intelligence and organization are learned properties.

This implementation allows researchers and enthusiasts to train a 25M parameter BDH model from scratch and provides the tools to "look inside its brain" to study its internal mechanisms.

## Features

-   **Faithful Implementation:** A clean, commented, and faithful implementation of the `BDHConfig`, `Attention`, and `BDH` classes based on the official source code.
-   **Advanced Training Script:** The `train.py` script uses modern, high-performance techniques including `torch.compile` (optional), Automatic Mixed Precision (`bfloat16`/`float16`), and efficient data loading.
-   **Reproducibility:** The project is structured for easy and reproducible environment setup using `uv`.
-   **Open-Source:** Licensed under the MIT license, encouraging community contribution and experimentation.

## Getting Started

This project uses `uv` for fast and reliable environment management. A GPU is highly recommended.

### 1. Clone the repository
```bash
git clone https://github.com/takzen/BDH-Official.git
cd BDH-Official
```

### 2. Install `uv` (if you don't have it)
```bash
pip install uv
```

### 3. Create environment and install dependencies
```bash
# Create the virtual environment
uv venv

# Activate the environment
# For Windows (PowerShell):
.\.venv\Scripts\activate
# For MacOS/Linux (bash/zsh):
source .venv/bin/activate

# Install all required packages
uv pip install -r requirements.txt
```

### 4. Run the training
The script will download the Tiny Shakespeare dataset and start training a 25M parameter model.
```bash
python train.py
```
*Note: The first run might be slow as `torch.compile` optimizes the model. Subsequent runs will be much faster.*

After training, the script will generate a sample text to demonstrate the model's capabilities.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.