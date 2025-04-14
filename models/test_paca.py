import torch
from paca import PACA

# Dummy dimensions
B, C, H, W = 2, 320, 32, 32  # Batch, Channels, Height, Width

# Create dummy UNet and ControlNet features
unet_feats = torch.randn(B, C, H, W, requires_grad=True)
controlnet_feats = torch.randn(B, C, H, W)

# Initialize PACA block
paca = PACA(embed_dim=C)

# Run PACA forward pass
print("Running PACA forward pass...")
out = paca(unet_feats, controlnet_feats)

# Verify output shape
assert out.shape == unet_feats.shape, f"Output shape mismatch: {out.shape} vs {unet_feats.shape}"
print("✅ Output shape is correct:", out.shape)

# Check gradients
loss = out.sum()
loss.backward()

assert unet_feats.grad is not None, "❌ No gradients passed through PACA"
print("✅ Gradients passed successfully")
print("🎉 PACA block test passed!")
