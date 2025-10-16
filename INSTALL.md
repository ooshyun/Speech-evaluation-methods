# Installation Guide

## Installing the Speech Evaluation Methods Package

This package can be installed in multiple ways:

### Method 1: Install in Development Mode (Recommended for Development)

This method creates a symlink to the source code, so changes you make are immediately reflected:

```bash
# From the project root directory
pip install -e .
```

### Method 2: Install with Dependencies

Install the package with all dependencies:

```bash
pip install -e .
```

### Method 3: Install from requirements.txt

If you prefer to install dependencies separately:

```bash
pip install -r requirements.txt
```

### Method 4: Using pyproject.toml (Modern Python)

```bash
pip install -e .
```

The `pyproject.toml` file will automatically handle dependencies.

## Verifying Installation

After installation, you can verify it works by running:

```python
import src
from src import estimate_stoi, estimate_pesq, estimate_snr

# Or import the package
import speech_evaluation_methods
```

Or test with the included test script:

```bash
python main.py
```

## Development Installation

If you want to contribute or modify the code:

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

This will install additional tools like `black` for code formatting and `pytest` for testing.

## Uninstalling

```bash
pip uninstall speech-evaluation-methods
```

## Dependencies

The main dependencies are:
- numpy>=1.21.0
- scipy>=1.7.0
- soundfile>=0.10.0
- librosa>=0.9.0
- museval>=0.4.0
- pystoi>=0.3.0
- pesq>=0.0.4
- torch>=1.11.0
- torchaudio>=0.11.0
- torchmetrics>=0.8.0
- resampy>=0.2.0
- auraloss

## Troubleshooting

### PyTorch Installation

If you have issues with PyTorch, you may need to install it separately first:

```bash
# For CPU-only (smaller download)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install the package:

```bash
pip install -e .
```

### PESQ Library Issues

If you encounter issues with the PESQ library on your system, you may need to install additional dependencies:

```bash
# On Ubuntu/Debian
sudo apt-get install build-essential

# On macOS (requires Xcode Command Line Tools)
xcode-select --install
```

