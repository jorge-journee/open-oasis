pip install -r requirements.txt
huggingface-cli login
huggingface-cli download Etched/oasis-500m oasis500m.pt # DiT checkpoint
huggingface-cli download Etched/oasis-500m vit-l-20.pt  # ViT VAE checkpoint