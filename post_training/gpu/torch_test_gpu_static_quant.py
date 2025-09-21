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

# For GPU quantization, we'll use FakeQuantize approach since native quantization is CPU-only
print("Setting up GPU-compatible quantization using FakeQuantize...")

# Create a CPU model for static quantization
cpu_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
cpu_model.fc = torch.nn.Linear(cpu_model.fc.in_features, 10)
cpu_model.eval()

# Get sample input for calibration
sample_input = next(iter(testing_dataloader))[0]

# Apply static quantization on CPU
cpu_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(cpu_model, inplace=True)

# Calibrate the model with some data
print("Calibrating quantized model...")
calib_count = 0
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(testing_dataloader):
        if calib_count >= 100:  # Use 100 samples for calibration
            break
        cpu_model(images)
        calib_count += images.size(0)

# Convert to quantized model
quantized_model = torch.quantization.convert(cpu_model, inplace=False)

# For GPU comparison, we'll also create a model with FakeQuantize
fake_quant_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
fake_quant_model.fc = torch.nn.Linear(fake_quant_model.fc.in_features, 10)
fake_quant_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
fake_quant_prepared = torch.quantization.prepare_qat(fake_quant_model, inplace=False)
fake_quant_prepared.to(device)
fake_quant_prepared.eval()

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
try:
    orig_params = sum(p.numel() for p in model.parameters())
    quant_params = sum(p.numel() for p in quantized_model.parameters())
    fake_quant_params = sum(p.numel() for p in fake_quant_prepared.parameters())
    print(f"Original model: {orig_params:,} parameters")
    print(f"Static quantized model: {quant_params:,} parameters")
    print(f"FakeQuantize model: {fake_quant_params:,} parameters")

    orig_size = get_model_size(model)
    quant_size = get_model_size(quantized_model)
    fake_quant_size = get_model_size(fake_quant_prepared)
    print(f"Original model file size: {orig_size:.2f} MB")
    print(f"Static quantized model file size: {quant_size:.2f} MB")
    print(f"FakeQuantize model file size: {fake_quant_size:.2f} MB")
    print(f"Static quantization size reduction: {((orig_size - quant_size) / orig_size * 100):.1f}%")

    orig_memory = get_model_memory_usage(model)
    quant_memory = get_model_memory_usage(quantized_model)
    fake_quant_memory = get_model_memory_usage(fake_quant_prepared)
    print(f"Original model memory usage: {orig_memory:.2f} MB")
    print(f"Static quantized model memory usage: {quant_memory:.2f} MB")
    print(f"FakeQuantize model memory usage: {fake_quant_memory:.2f} MB")
    print(f"Static quantization memory reduction: {((orig_memory - quant_memory) / orig_memory * 100):.1f}%")
except Exception as e:
    print(f"Note: Memory measurement failed: {e}")

print("\nTesting original (GPU) vs static quantized (CPU) vs FakeQuantize (GPU) models...")
acc_orig = 0
acc_quant = 0
acc_fake_quant = 0
total_orig_time = 0
total_quant_time = 0
total_fake_quant_time = 0
sample_count = 0

# Warm up GPU
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    _ = model(dummy_input)
    _ = fake_quant_prepared(dummy_input)
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

        # FakeQuantize model inference with timing (GPU)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        fake_quant_output = fake_quant_prepared(images_gpu)
        torch.cuda.synchronize()
        fake_quant_time = time.perf_counter() - start_time
        total_fake_quant_time += fake_quant_time

        # GPU memory after inference
        gpu_mem_after = get_gpu_memory_usage()

        # Static quantized model inference with timing (CPU)
        start_time = time.perf_counter()
        quantized_output = quantized_model(images)  # CPU inference
        quant_time = time.perf_counter() - start_time
        total_quant_time += quant_time

        # Move GPU outputs to CPU for comparison
        original_output_cpu = original_output.cpu()
        fake_quant_output_cpu = fake_quant_output.cpu()

        # Compare outputs
        diff_static = torch.abs(original_output_cpu - quantized_output).mean()
        diff_fake = torch.abs(original_output_cpu - fake_quant_output_cpu).mean()
        batch_size = images.size(0)
        print(f"Batch {batch_idx+1} (samples {sample_count+1}-{sample_count+batch_size}): "
              f"Static diff = {diff_static:.6f}, FakeQ diff = {diff_fake:.6f}, "
              f"GPU: {orig_time*1000:.3f}ms, CPU static: {quant_time*1000:.3f}ms, "
              f"GPU FakeQ: {fake_quant_time*1000:.3f}ms, GPU mem: {gpu_mem_after-gpu_mem_before:.1f}MB")

        # Get predictions
        original_pred = torch.argmax(original_output_cpu, dim=1)
        quantized_pred = torch.argmax(quantized_output, dim=1)
        fake_quant_pred = torch.argmax(fake_quant_output_cpu, dim=1)

        # Count correct predictions for this batch
        acc_orig += (original_pred == labels).sum().item()
        acc_quant += (quantized_pred == labels).sum().item()
        acc_fake_quant += (fake_quant_pred == labels).sum().item()
        sample_count += batch_size

print(f"Original model (GPU) accuracy: {acc_orig}/{sample_count} ({acc_orig/sample_count*100:.1f}%)")
print(f"Static quantized model (CPU) accuracy: {acc_quant}/{sample_count} ({acc_quant/sample_count*100:.1f}%)")
print(f"FakeQuantize model (GPU) accuracy: {acc_fake_quant}/{sample_count} ({acc_fake_quant/sample_count*100:.1f}%)")
print(f"Average GPU inference time per batch: {total_orig_time/(batch_idx+1)*1000:.3f}ms")
print(f"Average CPU static quantized inference time per batch: {total_quant_time/(batch_idx+1)*1000:.3f}ms")
print(f"Average GPU FakeQuantize inference time per batch: {total_fake_quant_time/(batch_idx+1)*1000:.3f}ms")
print(f"GPU vs CPU static speedup: {total_quant_time/total_orig_time:.2f}x")
print(f"GPU vs GPU FakeQuantize speedup: {total_fake_quant_time/total_orig_time:.2f}x")
print(f"Total GPU memory used: {get_gpu_memory_usage():.1f}MB")
print("GPU static quantization testing completed!")