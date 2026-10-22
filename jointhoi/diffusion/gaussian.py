import math

import torch
from torch import nn
from torch.nn import functional as F


def _cosine_betas(steps, max_beta=0.999):
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    values = []
    for index in range(steps):
        t1 = index / steps
        t2 = (index + 1) / steps
        values.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(values, dtype=torch.float32)


def _masked_mse(prediction, target, mask):
    per_frame = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
    mask = mask.to(per_frame.dtype)
    per_sample = (per_frame * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return per_sample.mean()


class GaussianDiffusion(nn.Module):
    """Clean-sample-prediction diffusion for joint motion/contact sequences."""

    def __init__(self, steps=1000, schedule="cosine", beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.steps = int(steps)
        if schedule == "cosine":
            betas = _cosine_betas(self.steps)
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.steps, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown noise schedule: {schedule}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    def training_losses(
        self,
        model,
        x_lhand,
        x_rhand,
        x_obj,
        x_dmap,
        condition,
        text,
        valid_lhand,
        valid_rhand,
        valid_obj,
        dmap_keep_mask=None,
    ):
        batch_size = x_lhand.shape[0]
        timestep = torch.randint(self.steps, (batch_size,), device=x_lhand.device)
        alpha = self.sqrt_alpha_bars[timestep].view(batch_size, 1, 1)
        sigma = self.sqrt_one_minus_alpha_bars[timestep].view(batch_size, 1, 1)

        def noise(sample):
            return alpha * sample + sigma * torch.randn_like(sample)

        predictions = model(
            noise(x_lhand),
            noise(x_rhand),
            noise(x_obj),
            noise(x_dmap),
            condition,
            timestep,
            text,
            valid_mask_lhand=valid_lhand,
            valid_mask_rhand=valid_rhand,
            valid_mask_obj=valid_obj,
        )
        pred_lhand, pred_rhand, pred_obj, pred_dmap = predictions[:4]
        split = x_dmap.shape[-1] // 2

        left_dmap_mask = valid_lhand
        right_dmap_mask = valid_rhand
        if dmap_keep_mask is not None:
            left_dmap_mask = left_dmap_mask & dmap_keep_mask[:, None]
            right_dmap_mask = right_dmap_mask & dmap_keep_mask[:, None]

        return {
            "lhand": _masked_mse(pred_lhand, x_lhand, valid_lhand),
            "rhand": _masked_mse(pred_rhand, x_rhand, valid_rhand),
            "object": _masked_mse(pred_obj, x_obj, valid_obj),
            "lhand_dmap": _masked_mse(pred_dmap[..., :split], x_dmap[..., :split], left_dmap_mask),
            "rhand_dmap": _masked_mse(pred_dmap[..., split:], x_dmap[..., split:], right_dmap_mask),
        }
