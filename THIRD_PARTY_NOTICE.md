Motion-context temporal anchor / payload patch logic in this build is adapted
from NikoDemon80/ComfyUI-H3-Motion-Context:
https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context

Upstream license: GPL-3.0.
The adaptation removes disk Save/Load and connects the previous sampled H3 latent
directly in RAM inside one ComfyUI DAG.

MiniMax H3 latent-upscaler architecture/normalization used by the optional
Latent Refine mode is compatible with the community model published by
LBH-123-AI:
https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler

Upstream model license: Apache-2.0.
The Extender downloads only the fixed BF16 SafeTensors checkpoint on first use
and verifies its SHA256 before loading it.

RelayVSR video super-resolution support (optional x2 post-processing) is based
on the native RelayVSR implementation in ComfyUI, using model checkpoints
repackaged by Comfy-Org:
https://github.com/kopperx/RelayVSR
https://huggingface.co/Comfy-Org/RelayVSR

RelayVSR was developed by Xijun Wang, Xin Li, Zirui Lang, Suhang Yao,
Haoran Li, and Zhibo Chen. The native x2 variant is credited to LightX2V:
https://huggingface.co/lightx2v/RelayVSR_X2

The original RelayVSR project and Comfy-Org model repository declare
Apache-2.0 licensing. Refer to the corresponding model repositories for the
terms applicable to the individual model checkpoints.

The Extender does not bundle RelayVSR weights or duplicate the RelayVSR
inference engine; it calls ComfyUI's native implementation and downloads the
Comfy-Org x2 model and FlashVAE checkpoints on demand if missing.
