Motion-context temporal anchor / payload patch logic in this build is adapted
from NikoDemon80/ComfyUI-H3-Motion-Context:
https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context

Upstream license: GPL-3.0.
The adaptation removes disk Save/Load and connects the previous sampled H3 latent
directly in RAM inside one ComfyUI DAG.

H3 3D latent upscaler network, normalization stats, and loading helpers in
`latent_upscaler.py` are adapted from LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler:
https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
Weights remain separate community artifacts under
`ComfyUI/models/latent_upscale_models/`.
