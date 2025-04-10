import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

# -------------------- CONFIG --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
INPUT_IMAGE_PATH = '/Volumes/Lenovo PS8/Casper/kaggle_dataset/TRAINCBCTSimulated2D/256/REC-1/slice_28.png'
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
# ------------------------------------------------

def load_cbct_image(path, resolution=512):
    """Load and preprocess a grayscale CBCT-style image."""
    print(f"📂 Loading image from: {path}")
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"❌ Could not read image at: {path}")

    image = cv2.resize(image, (resolution, resolution))
    image_rgb = np.stack([image] * 3, axis=-1)
    return Image.fromarray(image_rgb)



def main():
    print(f"🚀 Using device: {DEVICE} | torch_dtype: {DTYPE}")

    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL,
        torch_dtype=DTYPE
    )

    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=DTYPE
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if DEVICE == "cuda":
        pipe.enable_xformers_memory_efficient_attention()

    pipe = pipe.to(DEVICE)

    # Load conditioning image
    conditioning_image = load_cbct_image(INPUT_IMAGE_PATH)

    # Generation prompt
    prompt = (
        "a high-resolution axial CT scan of the lower abdomen, "
        "showing vertebrae, intestines with air pockets, and pelvic bones, "
        "grayscale medical imaging, DICOM-style"
    )
    negative_prompt = "sketch, painting, blurry, color, artistic"

    # Generate synthetic CT
    output = pipe(
        prompt=prompt,
        image=conditioning_image,
        num_inference_steps=30,
        guidance_scale=7.5,
        negative_prompt=negative_prompt
    )

    result = output.images[0]
    result_path = OUTPUT_DIR / "cbct_to_sct_result.png"
    result.save(result_path)
    print(f"✅ Output saved to {result_path}")


if __name__ == "__main__":
    main()
