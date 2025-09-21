import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import os
import pickle

# Check GPU availability and capabilities
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    # Check if Tensor Cores are available (requires compute capability >= 7.0 for FP16)
    capability = torch.cuda.get_device_capability()
    tensor_cores_available = capability[0] >= 7
    print(f"Tensor Cores available: {tensor_cores_available}")
else:
    tensor_cores_available = False

# Load a pre-trained model (using ResNet18 for CIFAR10)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes
model.to(device)
model.eval()

testing_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,  # Download data if needed
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
)

testing_dataloader = torch.utils.data.DataLoader(
    testing_dataset, batch_size=256, shuffle=False, num_workers=1  # Increased batch size for better GPU utilization
)

# Apply dynamic quantization (INT8) - note: dynamic quantization works on CPU model
print("Applying dynamic quantization...")
cpu_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
cpu_model.fc = torch.nn.Linear(cpu_model.fc.in_features, 10)
cpu_model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    cpu_model, {torch.nn.Linear}, dtype=torch.qint8
)

# For GPU quantization, try FP16 (half precision) which is natively supported
print("Creating GPU FP16 quantized model...")
gpu_quant_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
gpu_quant_model.fc = torch.nn.Linear(gpu_quant_model.fc.in_features, 10)
gpu_quant_model.to(device)
gpu_quant_model.half()  # Convert to FP16
gpu_quant_model.eval()
gpu_quantization_available = True
print("GPU FP16 quantization enabled")

def get_model_size(model):
    """Get model size in MB"""
    torch.save(model.state_dict(), 'temp_model.pth')
    size = os.path.getsize('temp_model.pth') / (1024 * 1024)  # Convert to MB
    os.remove('temp_model.pth')
    return size

def get_model_memory_usage(model):
    """Get model memory usage in MB"""
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0

print("Model size comparison:")
orig_params = sum(p.numel() for p in model.parameters())
quant_params = sum(p.numel() for p in quantized_model.parameters())
gpu_quant_params = sum(p.numel() for p in gpu_quant_model.parameters())
print(f"Original model (FP32): {orig_params:,} parameters")
print(f"CPU quantized model (INT8): {quant_params:,} parameters")
print(f"GPU quantized model (FP16): {gpu_quant_params:,} parameters")

orig_size = get_model_size(model)
quant_size = get_model_size(quantized_model)
gpu_quant_size = get_model_size(gpu_quant_model)
print(f"Original model file size: {orig_size:.2f} MB")
print(f"CPU quantized model file size: {quant_size:.2f} MB")
print(f"GPU quantized model file size: {gpu_quant_size:.2f} MB")
print(f"CPU quantization size reduction: {((orig_size - quant_size) / orig_size * 100):.1f}%")
print(f"GPU quantization size reduction: {((orig_size - gpu_quant_size) / orig_size * 100):.1f}%")

orig_memory = get_model_memory_usage(model)
quant_memory = get_model_memory_usage(quantized_model)
gpu_quant_memory = get_model_memory_usage(gpu_quant_model)
print(f"Original model memory usage: {orig_memory:.2f} MB")
print(f"CPU quantized model memory usage: {quant_memory:.2f} MB")
print(f"GPU quantized model memory usage: {gpu_quant_memory:.2f} MB")
print(f"CPU quantization memory reduction: {((orig_memory - quant_memory) / orig_memory * 100):.1f}%")
print(f"GPU quantization memory reduction: {((orig_memory - gpu_quant_memory) / orig_memory * 100):.1f}%")

print("\nTesting FP32 (GPU) vs INT8 (CPU) vs FP16 (GPU) models...")
acc_orig = 0
acc_quant = 0
acc_gpu_quant = 0
total_orig_time = 0
total_quant_time = 0
total_gpu_quant_time = 0
sample_count = 0

# Warm up GPU models
print("Warming up GPU models...")
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    dummy_input_half = dummy_input.half()
    
    # Warm up FP32 model
    for _ in range(5):
        _ = model(dummy_input)
        torch.cuda.synchronize()
    
    # Warm up FP16 model
    for _ in range(5):
        _ = gpu_quant_model(dummy_input_half)
        torch.cuda.synchronize()
    
print("Warmup completed.")

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(testing_dataloader):
        if sample_count >= 1024:  # Test 1024 samples (4 batches of 256)
            break

        # GPU memory before inference
        gpu_mem_before = get_gpu_memory_usage()

        # Original model inference with timing (GPU FP32) - includes data preparation
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        images_gpu = images.to(device)
        original_output = model(images_gpu)
        torch.cuda.synchronize()
        orig_time = time.perf_counter() - start_time
        total_orig_time += orig_time

        # GPU memory after inference
        gpu_mem_after = get_gpu_memory_usage()

        # Quantized model inference with timing (CPU)
        start_time = time.perf_counter()
        quantized_output = quantized_model(images)  # CPU inference
        quant_time = time.perf_counter() - start_time
        total_quant_time += quant_time

        # GPU FP16 model inference with timing - includes data preparation
        if gpu_quantization_available:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            images_gpu_half = images_gpu.half()  # Convert to FP16
            gpu_quant_output = gpu_quant_model(images_gpu_half)
            torch.cuda.synchronize()
            gpu_quant_time = time.perf_counter() - start_time
            total_gpu_quant_time += gpu_quant_time
            gpu_quant_output_cpu = gpu_quant_output.float().cpu()  # Convert back to FP32 for comparison
        else:
            gpu_quant_time = 0
            gpu_quant_output_cpu = original_output_cpu

        # Move original output to CPU for comparison
        original_output_cpu = original_output.cpu()

        # Compare outputs
        diff_cpu = torch.abs(original_output_cpu - quantized_output).mean()
        diff_gpu = torch.abs(original_output_cpu - gpu_quant_output_cpu).mean() if gpu_quantization_available else 0
        batch_size = images.size(0)
        
        if gpu_quantization_available:
            print(f"Batch {batch_idx+1} (samples {sample_count+1}-{sample_count+batch_size}): "
                  f"FP32: {orig_time*1000:.2f}ms, INT8: {quant_time*1000:.2f}ms, "
                  f"FP16: {gpu_quant_time*1000:.2f}ms, FP16 speedup: {orig_time/gpu_quant_time:.2f}x, "
                  f"GPU mem: {gpu_mem_after-gpu_mem_before:.1f}MB")
        else:
            print(f"Batch {batch_idx+1} (samples {sample_count+1}-{sample_count+batch_size}): "
                  f"FP32: {orig_time*1000:.2f}ms, INT8: {quant_time*1000:.2f}ms, "
                  f"GPU mem: {gpu_mem_after-gpu_mem_before:.1f}MB")

        # Get predictions
        original_pred = torch.argmax(original_output_cpu, dim=1)
        quantized_pred = torch.argmax(quantized_output, dim=1)
        if gpu_quantization_available:
            gpu_quant_pred = torch.argmax(gpu_quant_output_cpu, dim=1)

        # Count correct predictions for this batch
        acc_orig += (original_pred == labels).sum().item()
        acc_quant += (quantized_pred == labels).sum().item()
        if gpu_quantization_available:
            acc_gpu_quant += (gpu_quant_pred == labels).sum().item()
        sample_count += batch_size

print(f"\n=== RESULTS ===")
print(f"FP32 model (GPU) accuracy: {acc_orig}/{sample_count} ({acc_orig/sample_count*100:.1f}%)")
print(f"INT8 model (CPU) accuracy: {acc_quant}/{sample_count} ({acc_quant/sample_count*100:.1f}%)")
if gpu_quantization_available:
    print(f"FP16 model (GPU) accuracy: {acc_gpu_quant}/{sample_count} ({acc_gpu_quant/sample_count*100:.1f}%)")

print(f"\n=== PERFORMANCE ===")
print(f"Average FP32 (GPU) inference time per batch: {total_orig_time/(batch_idx+1)*1000:.2f}ms")
print(f"Average INT8 (CPU) inference time per batch: {total_quant_time/(batch_idx+1)*1000:.2f}ms")
if gpu_quantization_available:
    print(f"Average FP16 (GPU) inference time per batch: {total_gpu_quant_time/(batch_idx+1)*1000:.2f}ms")

print(f"\n=== SPEEDUP ANALYSIS ===")
if gpu_quantization_available:
    print(f"FP16 vs FP32 speedup: {total_orig_time/total_gpu_quant_time:.2f}x")
    print(f"FP16 vs INT8 speedup: {total_quant_time/total_gpu_quant_time:.2f}x")
print(f"INT8 vs FP32 speedup: {total_orig_time/total_quant_time:.2f}x (CPU vs GPU)")

print(f"\nTotal GPU memory used: {get_gpu_memory_usage():.1f}MB")
print("Quantization comparison completed!")