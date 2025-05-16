import torch
import time

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Get the name of the current GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Current GPU name: {gpu_name}")

    # Allocate a tensor on the GPU
    size = 8000
    start_time = time.time()
    a = torch.randn(size, size).cuda()
    b = torch.randn(size, size).cuda()
    c = torch.matmul(a, b)
    end_time = time.time()

    print(f"Matrix multiplication on GPU took: {end_time - start_time:.4f} seconds")

else:
    print("CUDA is not available. Please ensure you are running on a GPU-enabled node.")
