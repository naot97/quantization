# GPU Quantization Tests

This directory contains GPU-optimized versions of quantization tests for PyTorch models. All scripts test ResNet18 on CIFAR-10 dataset with consistent parameters (batch size 128, 512 test samples).

## Files Overview

### 1. `torch_test_gpu_dynamic_quant.py`
**Purpose**: Compares GPU baseline performance against CPU dynamic quantization

**Key Features**:
- GPU model inference with CUDA synchronization and timing
- CPU dynamic quantization using `torch.quantization.quantize_dynamic()`
- Memory usage tracking (GPU and model memory)
- Performance comparison between GPU and CPU quantized models

**Quantization Method**:
- Dynamic quantization (INT8) applied to Linear layers only
- Quantization runs on CPU, original model runs on GPU

**Metrics Tracked**:
- Model size reduction
- Memory usage reduction
- Inference time comparison (GPU vs CPU quantized)
- Accuracy comparison
- GPU memory consumption per batch

### 2. `torch_test_gpu_static_quant.py`
**Purpose**: Tests static quantization approaches for GPU compatibility

**Key Features**:
- Static quantization on CPU using `torch.quantization`
- FakeQuantize approach for GPU-compatible quantization
- Calibration using 100 samples from test dataset
- Three-way comparison: Original GPU vs Static CPU vs FakeQuantize GPU

**Quantization Methods**:
- **Static Quantization**: Full INT8 quantization with calibration (CPU only)
- **FakeQuantize**: Simulates quantization while keeping FP32 precision (GPU compatible)

**Metrics Tracked**:
- Model parameter counts and file sizes
- Memory usage for all three model variants
- Inference time comparison across all approaches
- Accuracy and output difference analysis

### 3. `torch_test_gpu_tensorrt.py`
**Purpose**: Advanced GPU optimization using NVIDIA TensorRT with INT8 quantization

**Key Features**:
- TensorRT compilation with multiple precision support (FP32, FP16, INT8)
- Entropy calibration for INT8 quantization
- Fallback mechanism if TensorRT compilation fails
- Comprehensive performance analysis

**Quantization Method**:
- TensorRT INT8 optimization with entropy calibration
- Uses DataLoaderCalibrator for automatic calibration
- Supports mixed precision (FP32/FP16/INT8)

**Advanced Features**:
- GPU memory tracking throughout inference
- TensorRT compilation error handling
- Batch processing optimization (128 samples per batch)
- Hardware-specific optimizations (GPU fallback enabled)

## Common Utilities

All scripts include shared utility functions:

### `get_model_size(model)`
Calculates model file size by saving state_dict and measuring file size.

### `get_model_memory_usage(model)`
Computes model memory footprint by summing parameter and buffer sizes.

### `get_gpu_memory_usage()`
Returns current GPU memory allocation in MB using `torch.cuda.memory_allocated()`.

## Testing Methodology

### Data Pipeline
- **Dataset**: CIFAR-10 test set (10,000 samples)
- **Batch Size**: 128 samples per batch
- **Test Samples**: 512 total (4 batches)
- **Preprocessing**: Standard CIFAR-10 normalization

### Performance Metrics
- **Accuracy**: Correct predictions / total predictions
- **Timing**: High-precision timing with GPU synchronization
- **Memory**: GPU memory allocation tracking
- **Model Size**: File size and parameter count analysis

### GPU Optimization
- **Warm-up**: Dummy inference before timing to initialize GPU
- **Synchronization**: `torch.cuda.synchronize()` for accurate timing
- **Memory Tracking**: Before/after memory measurement per batch

## Expected Outcomes

### Dynamic Quantization
- ~75% model size reduction
- Minimal accuracy loss (<1%)
- CPU quantized model slower than GPU baseline
- Significant memory savings

### Static Quantization
- Similar size reduction to dynamic quantization
- Better accuracy preservation than dynamic
- FakeQuantize shows GPU compatibility with minimal speedup
- CPU static quantization shows good compression

### TensorRT Optimization
- Potential 2-10x speedup depending on hardware
- Aggressive INT8 optimizations
- Hardware-specific acceleration
- May require calibration fine-tuning for accuracy

## Usage Notes

1. **GPU Requirements**: CUDA-capable GPU required for full functionality
2. **Dependencies**: PyTorch, TorchVision, TensorRT (for tensorrt script)
3. **Data**: Assumes CIFAR-10 data available in `./data` directory
4. **Memory**: Monitor GPU memory usage during execution
5. **Calibration**: TensorRT calibration cache saved to `./calibration.cache`

## Error Handling

- TensorRT compilation failures fallback to standard GPU inference
- Memory measurement failures handled gracefully
- GPU availability checked before CUDA operations
- Import errors provide informative messages