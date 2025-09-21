import torch
import torch_tensorrt
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Load a pre-trained model (using ResNet18 for CIFAR10)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes
model.eval()

testing_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
)

testing_dataloader = torch.utils.data.DataLoader(
    testing_dataset, batch_size=1, shuffle=False, num_workers=1
)

# Create calibrator for INT8 quantization
calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    testing_dataloader,
    cache_file="./calibration.cache",
    use_cache=False,
    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    device=torch.device("cuda:0"),
)

# Compile model with TensorRT
trt_mod = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 32, 32))],
    enabled_precisions={torch.float, torch.half, torch.int8},
    calibrator=calibrator,
    device={
        "device_type": torch_tensorrt.DeviceType.GPU,
        "gpu_id": 0,
        "dla_core": 0,
        "allow_gpu_fallback": False,
        "disable_tf32": False
    }
)

# Test the compiled model
print("Testing TensorRT compiled model...")
with torch.no_grad():
    for i, (images, labels) in enumerate(testing_dataloader):
        if i >= 10:  # Test only first 10 samples
            break

        images = images.cuda()

        # Original model inference
        original_output = model(images)

        # TensorRT model inference
        trt_output = trt_mod(images)

        # Compare outputs
        diff = torch.abs(original_output - trt_output).mean()
        print(f"Sample {i+1}: Mean absolute difference = {diff:.6f}")

        # Get predictions
        original_pred = torch.argmax(original_output, dim=1)
        trt_pred = torch.argmax(trt_output, dim=1)

        print(f"Original prediction: {original_pred.item()}, TensorRT prediction: {trt_pred.item()}, Ground truth: {labels.item()}")

print("Model testing completed!")