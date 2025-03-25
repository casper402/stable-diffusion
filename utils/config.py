import yaml
import torch

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.is_available()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    return device

def load_config(device, path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    base_path = config["device_paths"][device.type]

    paths = {
        key: f"{base_path}/{subpath}"
        for key, subpath in config["data_paths"].items()
    }
    config["paths"] = paths
    config["train"]["batch_size"] = config["train"]["batch_sizes"][device.type]

    float_keys = ["learning_rate", "min_learning_rate", "fine_tune_lr", "min_fine_tune_lr", "weight_decay", "fine_tune_weight_decay"]
    for key in float_keys:
        config["train"][key] = float(config["train"][key])

    return config