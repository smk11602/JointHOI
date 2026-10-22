import clip
import torch


def load_clip(version="ViT-B/32", device="cuda"):
    model, _ = clip.load(version, device="cpu", jit=False)
    if torch.device(device).type == "cuda":
        clip.model.convert_weights(model)
    else:
        model.float()
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def encode_text(model, prompts, device="cuda"):
    tokens = clip.tokenize(prompts, truncate=True).to(device)
    with torch.no_grad():
        return model.encode_text(tokens).float()
