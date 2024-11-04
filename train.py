import torch
from dit import DiT_models
from vae import VAE_models
from torchvision.io import read_video
from utils import one_hot_actions, sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange

# Ensure CUDA is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load DiT checkpoint
ckpt = torch.load("oasis500m.pt")
model = DiT_models["DiT-S/2"]()
model.load_state_dict(ckpt, strict=False)
model = model.to(device).train()  # Set to training mode

# Load VAE checkpoint
vae_ckpt = torch.load("vit-l-20.pt")
vae = VAE_models["vit-l-20-shallow-encoder"]()
vae.load_state_dict(vae_ckpt)
vae = vae.to(device).eval()  # Set to evaluation mode

# Sampling parameters
B = 1  # Batch size
total_frames = 32
max_noise_level = 1000
ddim_noise_steps = 16
noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1).long().to(device)
noise_abs_max = 20
ctx_max_noise_idx = (ddim_noise_steps // 10) * 3

# Get input video and actions
video_id = "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001"
mp4_path = f"sample_data/{video_id}.mp4"
actions_path = f"sample_data/{video_id}.actions.pt"
video = read_video(mp4_path, pts_unit="sec")[0].float() / 255
actions = one_hot_actions(torch.load(actions_path))

# Select a segment from the video
offset = 100
video = video[offset:offset + total_frames].unsqueeze(0)  # Shape: (1, total_frames, H, W, 3)
actions = actions[offset:offset + total_frames].unsqueeze(0)  # Shape: (1, total_frames, action_dim)

# Move data to device
x = video.to(device)  # Shape: (B, T, H, W, C)
actions = actions.to(device)

# VAE encoding for all frames
scaling_factor = 0.07843137255
x_flat = rearrange(x, "b t h w c -> (b t) c h w")  # Shape: (B*T, C, H, W)
H, W = x_flat.shape[-2:]

with torch.no_grad():
    x_flat = vae.encode(x_flat * 2 - 1).mean * scaling_factor  # VAE encoding

# Reshape back to (B, T, C, H, W)
x_encoded = rearrange(
    x_flat,
    "(b t) (h w) c -> b t c h w",
    b=B,
    t=total_frames,
    h=H // vae.patch_size,
    w=W // vae.patch_size
).to('cpu')

# Get alphas for noise scheduling
betas = sigmoid_beta_schedule(max_noise_level).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
num_epochs = 10
n_prompt_frames = 1  # Number of prompt frames (context)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    # Iterate over frames from n_prompt_frames to total_frames
    for i in range(n_prompt_frames, total_frames):
        x_input = x_encoded[:, :i + 1]  # Input frames up to current frame
        x_input = x_input.to('cuda')
        actions_input = actions[:, :i + 1]  # Corresponding actions
        B, T, C, H, W = x_input.shape
        start_frame = max(0, i + 1 - model.max_frames)

        # Sample noise indices
        noise_idx = torch.randint(1, ddim_noise_steps + 1, (1,)).item()
        ctx_noise_idx = min(noise_idx, ctx_max_noise_idx)

        # Prepare noise levels for context and current frame
        t_ctx = torch.full(
            (B, T - 1),
            noise_range[ctx_noise_idx],
            dtype=torch.long,
            device=device
        )
        t = torch.full(
            (B, 1),
            noise_range[noise_idx],
            dtype=torch.long,
            device=device
        )
        t_next = torch.full(
            (B, 1),
            noise_range[noise_idx - 1],
            dtype=torch.long,
            device=device
        )
        t_next = torch.where(t_next < 0, t, t_next)
        t = torch.cat([t_ctx, t], dim=1)
        t_next = torch.cat([t_ctx, t_next], dim=1)

        # Sliding window
        x_curr = x_input[:, start_frame:]
        t = t[:, start_frame:]
        t_next = t_next[:, start_frame:]
        actions_curr = actions_input[:, start_frame:start_frame + x_curr.shape[1]]
        B, T_curr, C, H, W = x_curr.shape

        # Add noise to context frames
        ctx_noise = torch.randn_like(x_curr[:, :-1])
        ctx_noise = torch.clamp(ctx_noise, -noise_abs_max, +noise_abs_max)
        x_noisy = x_curr.clone()
        x_noisy[:, :-1] = (
            alphas_cumprod[t[:, :-1]].sqrt() * x_noisy[:, :-1] +
            (1 - alphas_cumprod[t[:, :-1]]).sqrt() * ctx_noise
        )

        # Add noise to the current frame
        noise = torch.randn_like(x_curr[:, -1:])
        noise = torch.clamp(noise, -noise_abs_max, +noise_abs_max)
        x_noisy[:, -1:] = (
            alphas_cumprod[t[:, -1:]].sqrt() * x_noisy[:, -1:] +
            (1 - alphas_cumprod[t[:, -1:]]).sqrt() * noise
        )

        # Model prediction
        v = model(x_noisy, t, actions_curr)

        # Compute loss (only on the current frame)
        loss = torch.nn.functional.mse_loss(v[:, -1:], noise)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        x_input = x_input.to('cpu')
        del x_input
        torch.cuda.empty_cache()
    avg_loss = epoch_loss / (total_frames - n_prompt_frames)
    print(f"Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'models/finetuned_model.pt')
print("Model saved to models/finetuned_model.pt.")
