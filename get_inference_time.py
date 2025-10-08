import json
import torch
import time
from tqdm import tqdm
import gc
import os
from pathlib import Path

from TFNet import TFNet, reparameterize_model


def benchmark_model(model, input_tensor, num_warmup=100, num_iterations=1000, device_type='cpu'):
    """
    Benchmark model inference time

    Args:
        model: Model to test
        input_tensor: Input tensor
        num_warmup: Number of warmup iterations
        num_iterations: Number of test iterations
        device_type: Device type ('cpu' or 'cuda')

    Returns:
        avg_time: Average inference time (ms)
        std_time: Inference time standard deviation (ms)
        fps: Frames per second (FPS)
    """
    model.eval()
    model = reparameterize_model(model)

    # Warmup
    with torch.no_grad():
        for _ in tqdm(range(num_warmup), desc=f"Running {num_warmup} warmup iterations..."):
            _ = model(input_tensor)

    # Synchronize GPU (if using GPU)
    if device_type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Formal testing
    timings = []
    with torch.no_grad():
        for _ in tqdm(range(num_iterations), desc=f"Running {num_iterations} test iterations..."):
            start_time = time.time()
            _ = model(input_tensor)

            # Synchronize GPU (if using GPU)
            if device_type == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            timings.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Calculate statistics
    timings_tensor = torch.tensor(timings)
    avg_time = timings_tensor.mean().item()
    std_time = timings_tensor.std().item()
    fps = 1000 / avg_time  # Frames per second

    return avg_time, std_time, fps


def main(device_type='cpu'):
    """
    Main function to execute model benchmarking

    Args:
        device_type: Device type ('cpu' or 'cuda')
    """
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
        torch.set_num_threads(12)  # Or set to your CPU core count

    batch_size = 1
    input_resolution = [batch_size, 3, 512, 512]
    input_tensor = torch.randn(*input_resolution).to(device)

    print(f"Input resolution: {input_resolution}")
    print(f"Test configuration: batch_size={batch_size}")

    nc = 2
    wids = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    base_model_name = 'TFNet'

    # Set test parameters based on device type
    if device_type == 'cpu':
        num_warmup = 50
        num_iterations = 500
        output_dir = "./infer/CPU"
    else:  # cuda
        num_warmup = 600
        num_iterations = 1500
        output_dir = "./infer/GPU"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for wid in wids:
        for se in [False]:
            model_name_full = f'{base_model_name}_{nc}_{wid}_{se}'
            if device_type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                gc.collect()

            model = TFNet(num_classes=nc, width=wid, use_se=se)
            model.to(device)

            avg_time, std_time, fps = benchmark_model(
                model, input_tensor, num_warmup, num_iterations, device_type
            )

            model_info = {
                "model_name": model_name_full,
                "input_resolution": input_resolution,
                "avg_inference_time(ms)": round(avg_time, 2),
                "std_inference_time(ms)": round(std_time, 2),
                "fps": round(fps, 2),
                "device": str(device),
                "test_iterations": num_iterations,
                "warmup_iterations": num_warmup,
            }

            # Add device-specific information
            if device_type == 'cpu':
                model_info["cpu_threads"] = torch.get_num_threads()

            # Save as JSON file
            json_filename = os.path.join(output_dir, f"{model_name_full}.json")
            os.makedirs(os.path.dirname(json_filename), exist_ok=True)

            with open(json_filename, 'w') as f:
                json.dump(model_info, f, indent=4)

            print(f"Test completed for {model_name_full}:")
            print(f"  Average inference time: {avg_time:.2f} ms")
            print(f"  Time standard deviation: {std_time:.2f} ms")
            print(f"  FPS: {fps:.2f} FPS")
            if device_type == 'cpu':
                print(f"  CPU threads: {torch.get_num_threads()}")

            del model
            if device_type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\nAll models tested on {device_type.upper()} completed!")


def lowercase_device_type(value):
    return value.lower()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Model Inference Benchmark')
    parser.add_argument('--device', type=lowercase_device_type, choices=['cpu', 'cuda', 'gpu'], default='gpu',
                        help='Choose running device: cpu or cuda (default: cpu)')
    args = parser.parse_args()
    device_arg = args.device.lower()

    if device_arg in ['cuda', 'gpu'] and not torch.cuda.is_available():
        print("Warning: CUDA device selected but CUDA is not available, will use CPU instead")
        device_arg = 'cpu'
    elif device_arg == 'gpu':
        device_arg = 'cuda'

    main(device_type=device_arg)
