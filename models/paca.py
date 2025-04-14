import torch
import torch.nn as nn
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PACA Module ---
class PACA(nn.Module):
    def __init__(self, embed_dim):
        super(PACA, self).__init__()
        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x, y):
        # x: UNet features [B, C, H, W]
        # y: ControlNet features [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        y_flat = y.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        Q = self.to_q(x_flat)
        K = self.to_k(y_flat)
        V = self.to_v(y_flat)

        attn_weights = torch.softmax(Q @ K.transpose(-1, -2) * self.scale, dim=-1)
        out = attn_weights @ V
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return x + out  # residual connection


# --- Modified forward in training ---
def apply_paca_to_controlnet_output(unet_features, controlnet_output, paca_blocks):
    # Apply PACA to each UNet block where ControlNet residual is added
    down = [paca(u_feat, c_feat) for u_feat, c_feat, paca in zip(unet_features['down'], controlnet_output.down_block_res_samples, paca_blocks['down'])]
    mid = paca_blocks['mid'](unet_features['mid'], controlnet_output.mid_block_res_sample)
    return down, mid


# --- Model loading and setup ---
def setup_models_with_paca():
    from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler

    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(DEVICE)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny").to(DEVICE)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Freeze UNet and VAE
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Unfreeze ControlNet
    controlnet.requires_grad_(True)

    # Create PACA modules for each relevant block
    paca_blocks = {
        'down': nn.ModuleList([PACA(embed_dim=320) for _ in range(4)]),  # adjust dims as needed
        'mid': PACA(embed_dim=640),
    }
    paca_blocks['down'].requires_grad_(True)
    paca_blocks['mid'].requires_grad_(True)

    return vae, unet, controlnet, scheduler, paca_blocks


# --- Save & Load PACA Blocks ---
def save_paca_blocks(paca_blocks, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(paca_blocks['mid'].state_dict(), os.path.join(save_dir, 'paca_mid.pt'))
    for i, block in enumerate(paca_blocks['down']):
        torch.save(block.state_dict(), os.path.join(save_dir, f'paca_down_{i}.pt'))


def load_paca_blocks(paca_blocks, load_dir):
    paca_blocks['mid'].load_state_dict(torch.load(os.path.join(load_dir, 'paca_mid.pt')))
    for i, block in enumerate(paca_blocks['down']):
        block.load_state_dict(torch.load(os.path.join(load_dir, f'paca_down_{i}.pt')))

def extract_unet_features(unet, latents, timesteps, encoder_hidden_states):
    intermediate_outputs = {"down": [], "mid": None}

    def hook_down(module, input, output):
        intermediate_outputs["down"].append(output[0])  # skip res_samples

    def hook_mid(module, input, output):
        intermediate_outputs["mid"] = output

    # Clear hooks (in case already registered)
    for h in getattr(unet, "_hooks", []):
        h.remove()
    unet._hooks = []

    for block in unet.down_blocks:
        h = block.register_forward_hook(hook_down)
        unet._hooks.append(h)

    h = unet.mid_block.register_forward_hook(hook_mid)
    unet._hooks.append(h)

    # Run forward once to populate hooks
    with torch.no_grad():
        _ = unet(latents, timesteps, encoder_hidden_states=encoder_hidden_states)

    return {
        "down": intermediate_outputs["down"],
        "mid": intermediate_outputs["mid"]
    }
