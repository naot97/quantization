import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import os
import pickle

# Load a pre-trained model (using ResNet18 for CIFAR10)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes
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

# Apply dynamic quantization (INT8)
print("Applying dynamic quantization...")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
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

print("Testing original vs dynamically quantized model on CPU...")
acc_orig = 0
acc_quant = 0
total_orig_time = 0
total_quant_time = 0
sample_count = 0
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(testing_dataloader):
        if sample_count >= 512:  # Test 512 samples (4 batches of 128)
            break

        # Original model inference with timing
        start_time = time.perf_counter()
        original_output = model(images)
        orig_time = time.perf_counter() - start_time
        total_orig_time += orig_time

        # Quantized model inference with timing
        start_time = time.perf_counter()
        quantized_output = quantized_model(images)
        quant_time = time.perf_counter() - start_time
        total_quant_time += quant_time
        
        # Compare outputs
        diff = torch.abs(original_output - quantized_output).mean()
        batch_size = images.size(0)
        print(f"Batch {batch_idx+1} (samples {sample_count+1}-{sample_count+batch_size}): "
              f"Mean absolute difference = {diff:.6f}, "
              f"Orig time: {orig_time*1000:.3f}ms, Quant time: {quant_time*1000:.3f}ms")

        # Get predictions
        original_pred = torch.argmax(original_output, dim=1)
        quantized_pred = torch.argmax(quantized_output, dim=1)

        # Count correct predictions for this batch
        acc_orig += (original_pred == labels).sum().item()
        acc_quant += (quantized_pred == labels).sum().item()
        sample_count += batch_size
            
print(f"Original model accuracy: {acc_orig}/{sample_count} ({acc_orig/sample_count*100:.1f}%)")
print(f"Quantized model accuracy: {acc_quant}/{sample_count} ({acc_quant/sample_count*100:.1f}%)")
print(f"Average original inference time per batch: {total_orig_time/(batch_idx+1)*1000:.3f}ms")
print(f"Average quantized inference time per batch: {total_quant_time/(batch_idx+1)*1000:.3f}ms")
print(f"Speedup: {total_orig_time/total_quant_time:.2f}x")
print("CPU dynamic quantization testing completed!")
