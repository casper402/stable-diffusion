from torchvision import transforms

def build_train_transform(target_size):
    return transforms.Compose([
        transforms.Pad((0, 64, 0, 64)),
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])