import torch
from tqdm.auto import tqdm

from jointhoi.guidance.contact_inner import apply_contact_inner_guidance


def _ddim_step(x_t, pred_x0, t, previous_t, alpha_bars, eta):
    alpha_t = alpha_bars[t]
    alpha_previous = alpha_bars[previous_t] if previous_t >= 0 else alpha_t.new_tensor(1.0)
    pred_noise = (x_t - alpha_t.sqrt() * pred_x0) / (1.0 - alpha_t).sqrt().clamp_min(1e-8)
    sigma = eta * torch.sqrt(
        ((1.0 - alpha_previous) / (1.0 - alpha_t).clamp_min(1e-8))
        * (1.0 - alpha_t / alpha_previous).clamp_min(0.0)
    )
    direction = (1.0 - alpha_previous - sigma.square()).clamp_min(0.0).sqrt() * pred_noise
    if previous_t < 0:
        return pred_x0
    return alpha_previous.sqrt() * pred_x0 + direction + sigma * torch.randn_like(x_t)


def sample_ddim(
    diffusion,
    model,
    condition,
    text,
    nframes,
    hand_features,
    object_features,
    dmap_features,
    valid_lhand,
    valid_rhand,
    valid_object,
    steps=100,
    eta=0.0,
    cfg_scale=1.0,
    seed=0,
    cig=None,
):
    device = condition.device
    generator = torch.Generator(device=device).manual_seed(seed)
    shapes = (
        (1, nframes, hand_features),
        (1, nframes, hand_features),
        (1, nframes, object_features),
        (1, nframes, dmap_features),
    )
    state = [torch.randn(shape, generator=generator, device=device) for shape in shapes]
    steps = int(steps)
    if not 1 <= steps <= diffusion.steps:
        raise ValueError(f"steps must be between 1 and {diffusion.steps}, got {steps}")
    timesteps = torch.linspace(
        diffusion.steps - 1,
        0,
        steps,
        dtype=torch.long,
    ).tolist()

    for index, timestep in enumerate(tqdm(timesteps, desc="DDIM sampling")):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        t = torch.full((1,), timestep, device=device, dtype=torch.long)
        with torch.no_grad():
            conditional = model(
                *state, condition, t, text,
                valid_mask_lhand=valid_lhand,
                valid_mask_rhand=valid_rhand,
                valid_mask_obj=valid_object,
            )[:4]
            if cfg_scale != 1.0:
                null_dmap = model.null_dmap.expand_as(state[3])
                unconditional = model(
                    state[0], state[1], state[2], null_dmap, condition, t, text,
                    valid_mask_lhand=valid_lhand,
                    valid_mask_rhand=valid_rhand,
                    valid_mask_obj=valid_object,
                )[:4]
                prediction = [
                    uncond + cfg_scale * (cond - uncond)
                    for cond, uncond in zip(conditional[:3], unconditional[:3])
                ] + [conditional[3]]
            else:
                prediction = list(conditional)

        if cig is not None and timestep < diffusion.steps * cig["start_ratio"]:
            prediction[0], prediction[1], _ = apply_contact_inner_guidance(
                prediction[0], prediction[1], prediction[2], prediction[3],
                **{key: value for key, value in cig.items() if key != "start_ratio"},
            )

        state = [
            _ddim_step(current, estimate, timestep, previous, diffusion.alpha_bars, eta)
            for current, estimate in zip(state, prediction)
        ]

    return {
        "lhand": state[0],
        "rhand": state[1],
        "object": state[2],
        "dmap": state[3],
    }
