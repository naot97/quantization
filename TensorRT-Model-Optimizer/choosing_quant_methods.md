# Choosing Quantization Methods

## Quick Guide

**Small batch (≤4):** Use weight-only quantization
- INT4 AWQ or INT4-FP8 AWQ

**Large batch (≥16):** Use weight + activation quantization
- FP8 (best accuracy)
- INT4-FP8 AWQ (if FP8 isn't fast enough)
- INT8 SQ or INT4 AWQ (for Ampere/older GPUs)

## Method Comparison

| Method | Small Batch | Large Batch | Accuracy | Size | Notes |
|--------|-------------|-------------|----------|------|-------|
| FP8 | Medium | Medium | Excellent | 50% | Ada/Hopper+ only |
| INT8 SmoothQuant | Medium | Medium | Good | 50% | Most GPUs |
| INT4 AWQ | High | Low | Good | 25% | Ampere+ |
| INT4-FP8 AWQ | High | Medium | Good | 25% | Ada/Hopper+ |

## Deployment
- **TensorRT/TensorRT-LLM:** All methods
- **GPU Support:** Check method notes above
- **Calibration:** Minutes (FP8/INT8) to tens of minutes (INT4 methods)