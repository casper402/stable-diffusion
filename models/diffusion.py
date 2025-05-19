import torch

class Diffusion:
    def __init__(self, device, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

    def sample_timesteps(self, batch_size, generator=None):
        return torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch_size,),
            generator=generator,
            device=self.device
        )
