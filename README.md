# Quantization Project

A PyTorch implementation of W8A16 (Weight 8-bit, Activation 16-bit) quantization for neural network linear layers with support for automatic model conversion and Hugging Face integration.

## Overview

This project demonstrates how to implement quantized linear layers that reduce memory usage by storing weights as 8-bit integers while maintaining activations in higher precision. This approach provides 4x memory savings for weights while preserving reasonable numerical accuracy. The project now includes utilities to automatically replace linear layers in existing models and demonstrates quantization on real Hugging Face models.

## Features

- **W8A16 Quantization**: 8-bit weights, 16/32-bit activations
- **Per-channel scaling**: Individual scaling factors for each output channel
- **PyTorch integration**: Compatible with standard PyTorch workflows
- **Memory efficient**: 4x reduction in weight storage
- **Automatic layer replacement**: Convert existing models to use quantized layers
- **Hugging Face compatibility**: Works with transformers models
- **Selective quantization**: Exclude specific layers (e.g., language model heads)

## Files

- `main.py`: Demo script with dummy and real model examples
- `quantizer.py`: Main implementation with W8A16LinearLayer class and utilities
- `quantization_explanation.md`: Detailed technical explanation
- `requirements.txt`: Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
import torch
from quantizer import W8A16LinearLayer

# Create quantized layer
layer = W8A16LinearLayer(in_features=4, out_features=8, bias=True)

# Generate weights to quantize
weights = torch.randn((8, 4), dtype=torch.float32)
layer.quantize(weights)

# Forward pass
input_tensor = torch.randn((2, 4), dtype=torch.float32)
output = layer(input_tensor)
```

### Running the Demo

**Basic quantized layer demo:**
```bash
python quantizer.py
```

**Full demo with model replacement and Hugging Face integration:**
```bash
python main.py
```

## How It Works

1. **Quantization**: Float32 weights are converted to int8 using per-channel scaling
2. **Storage**: Weights stored as int8, scales as float32/float16
3. **Forward Pass**: Weights are cast back to input dtype and scaled during computation
4. **Memory Savings**: 4x reduction in weight storage

## Technical Details

The quantization process:
- Calculates per-channel scaling factors: `scale = max_abs_weight / 127`
- Quantizes weights: `int8_weight = round(weight / scale)`
- Dequantizes during forward pass: `dequantized = int8_weight * scale`

See `quantization_explanation.md` for detailed technical explanation.

### Model Replacement Example

```python
import torch
from transformers import AutoModelForCausalLM
from quantizer import W8A16LinearLayer, replace_linear_with_target_and_quantize

# Load a model
model = AutoModelForCausalLM.from_pretrained("model_name")

# Replace all linear layers except language model head with quantized versions
replace_linear_with_target_and_quantize(model, W8A16LinearLayer, ["lm_head"])
```

## Requirements

- Python 3.7+
- PyTorch 2.8+
- Transformers 4.56+

## License

This project is for educational and research purposes.