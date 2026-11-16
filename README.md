# cond-diff

A conditional diffusion model template

## Overview

This project provides a template for conditional generation using the EDM framework. 

## Features

- EDM-based diffusion model architecture
- Conditional generation capabilities


## Installation

### Prerequisites

- Python 3.10+ (3.11 recommended)
- NVIDIA driver with CUDA 12.8 support for GPU training
- For **RTX 5090 (Blackwell)**: use the provided `environment.yml`, which installs **PyTorch 2.7+ with CUDA 12.8** (`sm_120`).

### Setup

1. Clone the repository:
```bash
git clone https://github.com/zezhongzhang/glocal-condiff-turb
cd glocal-condiff-turb
```

2. Install environment (recreate if upgrading from an older PyTorch 1.12 env):
```bash
conda env remove -n condiff  # only if updating an old env
conda env create -f environment.yml
conda activate condiff
```

Verify GPU + PyTorch:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```
Expected: CUDA `12.8`, capability `(12, 0)` on RTX 5090.


3. Install the package in development mode:
```bash
pip install -e .
```

## Usage
run code in `bash_cmd_test`
1. test code by running the `# test code runs` block line by line
2. test code by running the `# SQG demo` block line by line


## Configuration

The main configuration file is located at `config/main.yaml`. Key settings include:

- change runs/runs.yaml for different runs

