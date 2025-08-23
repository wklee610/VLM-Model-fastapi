import torch
import platform

class GPUManager:
    def __init__(self):
        pass

    def is_gpu_available(self) -> torch.device:
        system = platform.system()
        # MAC OS
        if system == "Darwin":
            if torch.backends.mps.is_available():
                print("Apple GPU (MPS) is available")
                return torch.device("mps")
            else:
                print("No Apple GPU availablem. Using CPU")
                return torch.device("cpu")

        # NVIDIA GPU
        elif system == "Linux" or system == "Windows":
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
                print(f"NVIDIA GPU(s) available: {device_names}")
                return torch.device("cuda")
            else:
                print("No NVIDIA GPU available. Using CPU")
                return torch.device("cpu")
        
        # etc OS
        else:
            print(f"Unknown system ({system}). Using CPU")
            return torch.device("cpu")

    def free_gpu_resources(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
