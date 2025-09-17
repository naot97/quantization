# PyTorch CPU Quantization Comparison

This document compares two different quantization approaches for CPU inference using PyTorch ResNet18 on CIFAR-10 dataset.

## Test Configuration

- **Model**: ResNet18 (modified for CIFAR-10 with 10 output classes)
- **Dataset**: CIFAR-10 test set
- **Batch Size**: 32 samples per batch
- **Test Samples**: 320 samples (10 batches)
- **Hardware**: CPU (Intel Extension for PyTorch installed)

## Quantization Methods Tested

### 1. Dynamic Quantization (`torch_test_cpu_dynamic_quant.py`)

**Implementation**: Standard PyTorch dynamic quantization
- Uses `torch.quantization.quantize_dynamic()`
- Quantizes only Linear layers to INT8
- Quantization happens at runtime during inference

**Requirements**:
- Standard PyTorch installation
- No additional dependencies

### 2. Static Quantization (`torch_test_cpu_quantized_fixed.py`)

**Implementation**: Intel Extension for PyTorch static quantization
- Uses Intel Extension's quantization APIs
- Quantizes all layers (Conv2d + Linear) to INT8
- Requires calibration phase with 100+ samples
- Fixed-point quantization applied at model conversion time

**Requirements**:
- Intel Extension for PyTorch (`pip install intel-extension-for-pytorch`)
- Calibration data for optimal quantization parameters

## Performance Results

### Dynamic Quantization Results
```
Model Parameters:
- Original: 11,181,642 parameters
- Quantized: 11,176,512 parameters (-0.05% reduction)

Performance Metrics:
- Average original inference time per batch: ~5.6ms
- Average quantized inference time per batch: ~6.0ms
- Speedup: 0.93x (slightly slower)
- Quantization accuracy: Maintains identical predictions
- Mean absolute difference: ~0.004-0.008 between outputs
```

### Static Quantization Results
```
Performance Metrics:
- Average original inference time per batch: ~6.0ms
- Average quantized inference time per batch: ~69.0ms
- Speedup: 0.09x (significantly slower)
- Quantization accuracy: Maintains identical predictions
- Mean absolute difference: ~0.01-0.03 between outputs
```

## Analysis & Conclusions

### Dynamic Quantization
✅ **Pros**:
- Easy to implement with standard PyTorch
- Minimal setup required
- Low quantization error
- Maintains prediction accuracy
- Small memory footprint reduction

❌ **Cons**:
- Only quantizes Linear layers (Conv2d layers remain FP32)
- Slight performance overhead (~7% slower)
- Runtime quantization adds computation

### Static Quantization (Intel Extension)
✅ **Pros**:
- Quantizes all layer types (Conv2d + Linear)
- Lower numerical precision error in theory
- Should provide better performance in optimized deployment

❌ **Cons**:
- Significant performance overhead in current implementation (~11x slower)
- Requires additional dependency (Intel Extension)
- Complex calibration process
- Potential debugging/profiling overhead

## Recommendations

### For Development & Prototyping
**Use Dynamic Quantization** because:
- Simple implementation
- Acceptable performance trade-off
- Standard PyTorch compatibility
- Good balance of accuracy vs. complexity

### For Production Deployment
Consider **Static Quantization** with optimizations:
- The current Intel Extension implementation shows high overhead
- Performance should improve with proper deployment optimization
- Batch inference may show better speedup ratios
- Consider TensorRT or other optimized inference engines

## Key Findings

1. **Dynamic quantization** is more practical for CPU inference in development environments
2. **Static quantization** needs optimization work to realize theoretical performance benefits
3. **Batch processing** (32 samples) provides stable timing measurements
4. **Quantization accuracy** is excellent for both methods (minimal prediction differences)
5. **Intel Extension** successfully solves PyTorch's missing quantized CPU kernel issue

## Usage Instructions

```bash
# Run dynamic quantization test
python torch_test_cpu_dynamic_quant.py

# Run static quantization test
python torch_test_cpu_quantized_fixed.py
```

Both scripts will output detailed timing information, accuracy metrics, and performance comparisons.