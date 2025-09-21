import torch
import torch_tensorrt
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import os

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

# Create calibrator for INT8 quantization (using smaller batch for calibration)
calib_dataloader = torch.utils.data.DataLoader(
    testing_dataset, batch_size=1, shuffle=False, num_workers=1
)

print("Creating TensorRT calibrator...")
calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    calib_dataloader,
    cache_file="./calibration.cache",
    use_cache=False,
    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    device=device,
)

print("Compiling model with TensorRT...")
try:
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input((128, 3, 32, 32))],  # Batch size 128
        enabled_precisions={torch.float, torch.half, torch.int8},
        calibrator=calibrator,
        device={
            "device_type": torch_tensorrt.DeviceType.GPU,
            "gpu_id": 0,
            "dla_core": 0,
            "allow_gpu_fallback": True,
            "disable_tf32": False
        }
    )
    trt_available = True
    print("TensorRT compilation successful!")
except Exception as e:
    print(f"TensorRT compilation failed: {e}")
    print("Proceeding with regular GPU inference only...")
    trt_available = False

print("Model information:")
params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params:,}")
model_size = get_model_size(model)
print(f"Model file size: {model_size:.2f} MB")
model_memory = get_model_memory_usage(model)
print(f"Model memory usage: {model_memory:.2f} MB")

print("\nTesting GPU model performance...")
acc_orig = 0
acc_trt = 0
total_orig_time = 0
total_trt_time = 0
sample_count = 0

# Warm up GPU
with torch.no_grad():
    dummy_input = torch.randn(128, 3, 32, 32).to(device)
    _ = model(dummy_input)
    if trt_available:
        _ = trt_model(dummy_input)
    torch.cuda.synchronize()

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(testing_dataloader):
        if sample_count >= 512:  # Test 512 samples (4 batches of 128)
            break

        # GPU memory before inference
        gpu_mem_before = get_gpu_memory_usage()

        # Original model inference with timing
        images_gpu = images.to(device)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        original_output = model(images_gpu)
        torch.cuda.synchronize()
        orig_time = time.perf_counter() - start_time
        total_orig_time += orig_time

        # TensorRT model inference with timing (if available)
        if trt_available:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            trt_output = trt_model(images_gpu)
            torch.cuda.synchronize()
            trt_time = time.perf_counter() - start_time
            total_trt_time += trt_time

            # Compare outputs
            diff = torch.abs(original_output - trt_output).mean()
        else:
            trt_time = 0
            diff = 0

        # GPU memory after inference
        gpu_mem_after = get_gpu_memory_usage()

        # Get predictions
        original_pred = torch.argmax(original_output, dim=1)
        correct_orig = (original_pred.cpu() == labels).sum().item()
        acc_orig += correct_orig

        if trt_available:
            trt_pred = torch.argmax(trt_output, dim=1)
            correct_trt = (trt_pred.cpu() == labels).sum().item()
            acc_trt += correct_trt

        batch_size = images.size(0)
        sample_count += batch_size

        if trt_available:
            print(f"Batch {batch_idx+1} (samples {sample_count-batch_size+1}-{sample_count}): "
                  f"Orig: {correct_orig}/{batch_size}, TRT: {correct_trt}/{batch_size}, "
                  f"Diff: {diff:.6f}, Orig time: {orig_time*1000:.3f}ms, "
                  f"TRT time: {trt_time*1000:.3f}ms, GPU mem: {gpu_mem_after-gpu_mem_before:.1f}MB")
        else:
            print(f"Batch {batch_idx+1} (samples {sample_count-batch_size+1}-{sample_count}): "
                  f"Accuracy: {correct_orig}/{batch_size} ({correct_orig/batch_size*100:.1f}%), "
                  f"Time: {orig_time*1000:.3f}ms, GPU mem: {gpu_mem_after-gpu_mem_before:.1f}MB")

print(f"\nOverall Results:")
print(f"Original model accuracy: {acc_orig}/{sample_count} ({acc_orig/sample_count*100:.1f}%)")
if trt_available:
    print(f"TensorRT model accuracy: {acc_trt}/{sample_count} ({acc_trt/sample_count*100:.1f}%)")
print(f"Average original inference time per batch: {total_orig_time/(batch_idx+1)*1000:.3f}ms")
if trt_available:
    print(f"Average TensorRT inference time per batch: {total_trt_time/(batch_idx+1)*1000:.3f}ms")
    print(f"TensorRT speedup: {total_orig_time/total_trt_time:.2f}x")
print(f"Average inference time per sample: {total_orig_time/sample_count*1000:.3f}ms")
print(f"Total GPU memory used: {get_gpu_memory_usage():.1f}MB")
print("GPU TensorRT testing completed!")