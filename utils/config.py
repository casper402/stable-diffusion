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

    if device.type == "cpu":
        config["train"]["batch_size"] = 1
        config["train"]["subset_size"] = 10

    train_float_keys = ["learning_rate", "min_learning_rate", "fine_tune_lr", "min_fine_tune_lr", "weight_decay", "fine_tune_weight_decay"]
    for key in train_float_keys:
        config["train"][key] = float(config["train"][key])

    vae_float_keys = ["beta", "lambda_perceptual"]
    for key in vae_float_keys:
        config["vae"][key] = float(config["vae"][key])
        print(f"{key}: ", config["vae"][key])

    return config