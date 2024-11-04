"""

PyGame implementation of open-oasis by Miklos Nagy: miklos.mnagy@gmail.com www.github.com/XmYx

References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""

import torch
from dit import DiT_models
from vae import VAE_models
from torchvision.io import read_video
from utils import sigmoid_beta_schedule
from einops import rearrange
from torch import autocast
import pygame
import numpy as np

assert torch.cuda.is_available()
device = "cuda:0"

# Define ACTION_KEYS
ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]

# Helper functions to capture live actions
def get_current_action():
    action = {}
    keys = pygame.key.get_pressed()
    mouse_buttons = pygame.mouse.get_pressed()
    mouse_rel = pygame.mouse.get_rel()  # returns (x, y)

    # Map keys to actions
    action["inventory"] = 1 if keys[pygame.K_e] else 0
    action["ESC"] = 1 if keys[pygame.K_ESCAPE] else 0
    action["hotbar.1"] = 1 if keys[pygame.K_1] else 0
    action["hotbar.2"] = 1 if keys[pygame.K_2] else 0
    action["hotbar.3"] = 1 if keys[pygame.K_3] else 0
    action["hotbar.4"] = 1 if keys[pygame.K_4] else 0
    action["hotbar.5"] = 1 if keys[pygame.K_5] else 0
    action["hotbar.6"] = 1 if keys[pygame.K_6] else 0
    action["hotbar.7"] = 1 if keys[pygame.K_7] else 0
    action["hotbar.8"] = 1 if keys[pygame.K_8] else 0
    action["hotbar.9"] = 1 if keys[pygame.K_9] else 0
    action["forward"] = 1 if keys[pygame.K_w] else 0
    action["back"] = 1 if keys[pygame.K_s] else 0
    action["left"] = 1 if keys[pygame.K_a] else 0
    action["right"] = 1 if keys[pygame.K_d] else 0
    action["camera"] = mouse_rel  # tuple (x, y)
    action["jump"] = 1 if keys[pygame.K_SPACE] else 0
    action["sneak"] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
    action["sprint"] = 1 if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL] else 0
    action["swapHands"] = 0  # Map to a key if needed
    action["attack"] = 1 if mouse_buttons[0] else 0  # Left mouse button
    action["use"] = 1 if mouse_buttons[2] else 0     # Right mouse button
    action["pickItem"] = 0  # Map to a key if needed
    action["drop"] = 1 if keys[pygame.K_q] else 0

    return action

def action_to_tensor(action):
    actions_one_hot = torch.zeros(1, len(ACTION_KEYS), device=device)
    for j, action_key in enumerate(ACTION_KEYS):
        if action_key.startswith("camera"):
            if action_key == "cameraX":
                value = action["camera"][0]
            elif action_key == "cameraY":
                value = action["camera"][1]
            else:
                raise ValueError(f"Unknown camera action key: {action_key}")
            # Normalize value to be in [-1, 1]
            max_val = 20
            bin_size = 0.5
            num_buckets = int(max_val / bin_size)
            value = (value) / num_buckets
            value = max(min(value, 1.0), -1.0)
        else:
            value = action.get(action_key, 0)
            value = float(value)
        actions_one_hot[0, j] = value
    return actions_one_hot

# Initialize pygame
pygame.init()
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

# Set up display
screen_width = 640  # Adjust as needed
screen_height = 360  # Adjust as needed
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Generated Video")

# Load DiT checkpoint
ckpt = torch.load("oasis500m.pt")
model = DiT_models["DiT-S/2"]()
model.load_state_dict(ckpt, strict=False)
model = model.to(device).eval()

# Load VAE checkpoint
vae_ckpt = torch.load("vit-l-20.pt")
vae = VAE_models["vit-l-20-shallow-encoder"]()
vae.load_state_dict(vae_ckpt)
vae = vae.to(device).eval()

# Sampling params
B = 1
total_frames = 10000  # Run indefinitely or set as needed
max_noise_level = 1000
ddim_noise_steps = 100
noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
noise_abs_max = 20
ctx_max_noise_idx = ddim_noise_steps // 10 * 3

# Get input video (first frame as prompt)
video_id = "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001"
mp4_path = f"sample_data/{video_id}.mp4"
video = read_video(mp4_path, pts_unit="sec")[0].float() / 255

offset = 100
video = video[offset:]
n_prompt_frames = 1
x = video[:n_prompt_frames].unsqueeze(0).to(device)

# VAE encoding
scaling_factor = 0.07843137255
x = rearrange(x, "b t h w c -> (b t) c h w")
H, W = x.shape[-2:]
with torch.no_grad():
    x = vae.encode(x * 2 - 1).mean * scaling_factor
x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H//vae.patch_size, w=W//vae.patch_size)

# Get alphas
betas = sigmoid_beta_schedule(max_noise_level).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

# Initialize action list
actions_list = []

# Main loop
clock = pygame.time.Clock()
running = True
i = n_prompt_frames

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture current action
    action = get_current_action()
    actions_curr = action_to_tensor(action).unsqueeze(0)  # Shape [1, num_actions]
    #actions_list = [action_tensor]

    # Generate a random latent for the new frame
    chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
    chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
    x = torch.cat([x, chunk], dim=1)
    start_frame = max(0, i + 1 - model.max_frames)

    # Prepare actions for current context
    #actions_curr = torch.cat(actions_list[start_frame:i+1], dim=0).unsqueeze(0)  # Shape [1, context_length, num_actions]
    print(actions_curr.shape)
    for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
        # Set up noise values
        ctx_noise_idx = min(noise_idx, ctx_max_noise_idx)
        t_ctx  = torch.full((B, i), noise_range[ctx_noise_idx], dtype=torch.long, device=device)
        t      = torch.full((B, 1), noise_range[noise_idx],     dtype=torch.long, device=device)
        t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
        t_next = torch.where(t_next < 0, t, t_next)
        t = torch.cat([t_ctx, t], dim=1)
        t_next = torch.cat([t_ctx, t_next], dim=1)

        # Sliding window
        x_curr = x[:, start_frame:].clone()
        t = t[:, start_frame:]
        t_next = t_next[:, start_frame:]

        # Add noise to context frames
        if x_curr.shape[1] > 1:
            ctx_noise = torch.randn_like(x_curr[:, :-1])
            ctx_noise = torch.clamp(ctx_noise, -noise_abs_max, +noise_abs_max)
            x_curr[:, :-1] = alphas_cumprod[t[:, :-1]].sqrt() * x_curr[:, :-1] + \
                             (1 - alphas_cumprod[t[:, :-1]]).sqrt() * ctx_noise

        # Get model predictions
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                v = model(x_curr, t, actions_curr)

        x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
        x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / \
                  (1 / alphas_cumprod[t] - 1).sqrt()

        # Get frame prediction
        x_pred = alphas_cumprod[t_next].sqrt() * x_start + x_noise * (1 - alphas_cumprod[t_next]).sqrt()
        x[:, -1:] = x_pred[:, -1:]

    # VAE decoding of the last frame
    x_last = x[:, -1:]
    x_last = rearrange(x_last, "b t c h w -> (b t) (h w) c")
    with torch.no_grad():
        x_decoded = (vae.decode(x_last / scaling_factor) + 1) / 2
    x_decoded = rearrange(x_decoded, "(b t) c h w -> b t h w c", b=1, t=1)
    x_decoded = torch.clamp(x_decoded, 0, 1)
    x_decoded = (x_decoded * 255).byte().cpu().numpy()
    frame = x_decoded[0, 0]

    # Convert to surface and display
    frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    frame_surface = pygame.transform.scale(frame_surface, (screen_width, screen_height))
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()

    # Increment frame index
    i += 1

    # Control frame rate
    clock.tick(1)  # Adjust FPS as needed

pygame.quit()
