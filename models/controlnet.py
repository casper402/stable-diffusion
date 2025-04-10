from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

def load_pipeline(controlnet_model_path=None):
    base_model = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if controlnet_model_path:
        raise NotImplementedError("ControlNet not set up in this minimal version.")

    print(f"🧪 Loading Stable Diffusion on {device}...")

    # Only use float16 if on GPU
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    pipe = pipe.to(device)
    return pipe

def generate_image(pipe, prompt="synthetic CT scan, axial abdominal slice, realistic medical contrast, DICOM-like scan, clean edges, noise-free"):
    image = pipe(prompt=prompt, negative_prompt="sketch, painting, blurry, artistic, color").images[0]
    output_path = Path("outputs/generated_sample3.png")
    output_path.parent.mkdir(exist_ok=True)
    image.save(output_path)
    print(f"✅ Image saved to {output_path}")

if __name__ == "__main__":
    pipe = load_pipeline()
    generate_image(pipe)
