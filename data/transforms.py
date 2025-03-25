from torchvision import transforms
from torchvision.transforms import functional as F

class ResizeAndPad:
    def __init__(self, target_size):
        self.target_size = target_size
    
    def __call__(self, image):
        w, h = image.size
        aspect = w / h

        if aspect > 1:
            new_w = self.target_size
            new_h = int(self.target_size / aspect)
        else:
            new_h = self.target_size
            new_w = int(self.target_size * aspect)

        image = F.resize(image, (new_w, new_h), interpolation=transforms.InterpolationMode.LANCZOS)
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        padding = (pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2)
        image = F.pad(image, padding, fill=0)

        return image

def build_train_transform(target_size):
    return transforms.Compose([
        ResizeAndPad(target_size),
        transforms.ToTensor(),
    ])