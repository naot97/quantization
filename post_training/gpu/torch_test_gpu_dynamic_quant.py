import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import os
import pickle

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a pre-trained model (using ResNet18 for CIFAR10)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes
model.to(device)
model.eval()

testing_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,  # Data already downloaded
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
)

testing_dataloader = torch.utils.data.DataLoader(
    testing_dataset, batch_size=128, shuffle=False, num_workers=1
)

# Apply dynamic quantization (INT8) - note: dynamic quantization works on CPU model
print("Applying dynamic quantization...")
cpu_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
cpu_model.fc = torch.nn.Linear(cpu_model.fc.in_features, 10)
cpu_model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    cpu_model, {torch.nn.Linear}, dtype=torch.qint8
)

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
print(f"Original model: {orig_params:,} parameters")
print(f"Quantized model: {quant_params:,} parameters")

orig_size = get_model_size(model)
quant_size = get_model_size(quantized_model)
print(f"Original model file size: {orig_size:.2f} MB")
print(f"Quantized model file size: {quant_size:.2f} MB")
print(f"Size reduction: {((orig_size - quant_size) / orig_size * 100):.1f}%")

orig_memory = get_model_memory_usage(model)
quant_memory = get_model_memory_usage(quantized_model)
print(f"Original model memory usage: {orig_memory:.2f} MB")
print(f"Quantized model memory usage: {quant_memory:.2f} MB")
print(f"Memory reduction: {((orig_memory - quant_memory) / orig_memory * 100):.1f}%")

print("\nTesting original (GPU) vs dynamically quantized (CPU) model...")
acc_orig = 0
acc_quant = 0
total_orig_time = 0
total_quant_time = 0
sample_count = 0

# Warm up GPU
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    _ = model(dummy_input)
    torch.cuda.synchronize()

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(testing_dataloader):
        if sample_count >= 512:  # Test 512 samples (4 batches of 128)
            break

        # GPU memory before inference
        gpu_mem_before = get_gpu_memory_usage()

        # Original model inference with timing (GPU)
        images_gpu = images.to(device)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
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

        # Move original output to CPU for comparison
        original_output_cpu = original_output.cpu()

        # Compare outputs
        diff = torch.abs(original_output_cpu - quantized_output).mean()
        batch_size = images.size(0)
        print(f"Batch {batch_idx+1} (samples {sample_count+1}-{sample_count+batch_size}): "
              f"Mean absolute difference = {diff:.6f}, "
              f"GPU time: {orig_time*1000:.3f}ms, CPU quant time: {quant_time*1000:.3f}ms, "
              f"GPU mem: {gpu_mem_after-gpu_mem_before:.1f}MB")

        # Get predictions
        original_pred = torch.argmax(original_output_cpu, dim=1)
        quantized_pred = torch.argmax(quantized_output, dim=1)

        # Count correct predictions for this batch
        acc_orig += (original_pred == labels).sum().item()
        acc_quant += (quantized_pred == labels).sum().item()
        sample_count += batch_size

print(f"Original model (GPU) accuracy: {acc_orig}/{sample_count} ({acc_orig/sample_count*100:.1f}%)")
print(f"Quantized model (CPU) accuracy: {acc_quant}/{sample_count} ({acc_quant/sample_count*100:.1f}%)")
print(f"Average GPU inference time per batch: {total_orig_time/(batch_idx+1)*1000:.3f}ms")
print(f"Average CPU quantized inference time per batch: {total_quant_time/(batch_idx+1)*1000:.3f}ms")
print(f"GPU vs CPU quant speedup: {total_quant_time/total_orig_time:.2f}x")
print(f"Total GPU memory used: {get_gpu_memory_usage():.1f}MB")
print("GPU vs CPU dynamic quantization testing completed!")