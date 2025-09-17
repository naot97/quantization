import torch
import intel_extension_for_pytorch as ipex
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time

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
    testing_dataset, batch_size=32, shuffle=False, num_workers=1
)

# Get sample input
sample_input = next(iter(testing_dataloader))[0]

# Use Intel Extension for proper static quantization
print("Applying Intel Extension static quantization...")
qconfig_mapping = ipex.quantization.default_static_qconfig_mapping
prepared_model = ipex.quantization.prepare(model, qconfig_mapping, sample_input, inplace=False)

# Calibrate the model with some data
print("Calibrating quantized model...")
calib_count = 0
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(testing_dataloader):
        if calib_count >= 100:  # Use 100 samples for calibration
            break
        prepared_model(images)
        calib_count += images.size(0)

# Convert to quantized model
quantized_model = ipex.quantization.convert(prepared_model)

print("Testing original vs Intel static quantized model on CPU...")
acc_orig = 0
acc_quant = 0
total_orig_time = 0
total_quant_time = 0
sample_count = 0
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(testing_dataloader):
        if sample_count >= 320:  # Test 320 samples (10 batches of 32)
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
print("Intel Extension CPU static quantization testing completed!")
