import torch
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
    testing_dataset, batch_size=8, shuffle=False, num_workers=1
)

# Apply dynamic quantization (INT8)
print("Applying dynamic quantization...")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

print("Model size comparison:")
print(f"Original model: {sum(p.numel() for p in model.parameters())} parameters")
print(f"Quantized model: {sum(p.numel() for p in quantized_model.parameters())} parameters")

print("Testing original vs dynamically quantized model on CPU...")
acc_orig = 0
acc_quant = 0
total_orig_time = 0
total_quant_time = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(testing_dataloader):
        if i >= 10:  # Test only first 10 samples
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
        print(f"Sample {i+1}: Mean absolute difference = {diff:.6f}, Orig time: {orig_time*1000:.3f}ms, Quant time: {quant_time*1000:.3f}ms")

        # Get predictions
        original_pred = torch.argmax(original_output, dim=1)
        quantized_pred = torch.argmax(quantized_output, dim=1)

        print(f"Original prediction: {original_pred.item()}, Quantized prediction: {quantized_pred.item()}, Ground truth: {labels.item()}")
        if original_pred.item() == labels.item():
            acc_orig += 1
            
        if quantized_pred.item() == labels.item():
            acc_quant += 1
            
print(f"Original model accuracy (first 10 samples): {acc_orig}/10")
print(f"Quantized model accuracy (first 10 samples): {acc_quant}/10")
print(f"Average original inference time: {total_orig_time/10*1000:.3f}ms")
print(f"Average quantized inference time: {total_quant_time/10*1000:i chan.3f}ms")
print(f"Speedup: {total_orig_time/total_quant_time:.2f}x")
print("CPU dynamic quantization testing completed!")
