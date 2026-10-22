import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from jointhoi.data.dataset import normalize_point_cloud
from jointhoi.data.objects import ObjectRepository
from jointhoi.diffusion import GaussianDiffusion
from jointhoi.diffusion.sampling import sample_ddim
from jointhoi.models import JointHOI, PointNetfeat, build_mano
from jointhoi.models.text import encode_text, load_clip


def parse_args():
    parser = argparse.ArgumentParser(description="Sample a JointHOI sequence")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--object", required=True, dest="object_name")
    parser.add_argument("--hands", choices=("left", "right", "both"), default="both")
    parser.add_argument("--nframes", type=int)
    parser.add_argument("--output", default="outputs/sample.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-guidance", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def load_config(path, overrides):
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    return config


def load_state(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)


def load_pointnet(path, device):
    model = PointNetfeat(global_feat=False, feature_transform=False, in_dim=3).to(device)
    load_state(model, path, device)
    return model.eval().requires_grad_(False)


def main():
    args = parse_args()
    config = load_config(args.config, args.overrides)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    nframes = args.nframes or config.dataset.max_nframes
    if nframes > config.dataset.max_nframes:
        raise ValueError(f"nframes must be <= {config.dataset.max_nframes}")

    is_lhand = args.hands in {"left", "both"}
    is_rhand = args.hands in {"right", "both"}
    objects = ObjectRepository(config.dataset.object_path)
    object_data = objects.get(args.object_name)
    normalized, centroid, scale = normalize_point_cloud(object_data["points"])

    pointnet = load_pointnet(config.pointnet_checkpoint, device)
    points = torch.from_numpy(normalized).float().unsqueeze(0).to(device)
    with torch.no_grad():
        features = pointnet(points)
        if features.ndim == 3:
            features = features[:, 0]
    condition = torch.cat(
        [
            features,
            torch.tensor([[scale]], device=device),
            torch.from_numpy(centroid).float().unsqueeze(0).to(device),
        ],
        dim=-1,
    )
    clip_model = load_clip(config.clip_version, device)
    text = encode_text(clip_model, [args.prompt], device)

    model = JointHOI(
        hand_nfeats=99,
        obj_nfeats=config.model.obj_nfeats,
        dmap_nfeats=2048,
        cond_dim=1092,
        latent_dim=config.model.latent_dim,
        ff_size=config.model.ff_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        max_nframes=config.dataset.max_nframes,
        num_train_timesteps=config.diffusion.steps,
    ).to(device)
    load_state(model, args.checkpoint, device)
    model.eval()
    diffusion = GaussianDiffusion(
        steps=config.diffusion.steps,
        schedule=config.diffusion.schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
    ).to(device)

    valid_lhand = torch.full((1, nframes), is_lhand, dtype=torch.bool, device=device)
    valid_rhand = torch.full((1, nframes), is_rhand, dtype=torch.bool, device=device)
    valid_object = torch.ones((1, nframes), dtype=torch.bool, device=device)

    with np.load(config.dataset.stats_path) as raw_stats:
        stats = {name: float(raw_stats[name]) for name in (
            "ldmap_min", "ldmap_max", "rdmap_min", "rdmap_max"
        )}

    cig = None
    if config.sampling.guidance.enabled and not args.no_guidance:
        lhand_layer = build_mano(config.mano_path, False, config.dataset.flat_hand).to(device)
        rhand_layer = build_mano(config.mano_path, True, config.dataset.flat_hand).to(device)
        top_mask = object_data["top_mask"]
        if top_mask is not None:
            top_mask = torch.from_numpy(top_mask).to(device)
        cig = {
            "object_points": torch.from_numpy(object_data["points"]).float().to(device),
            "stats": stats,
            "lhand_layer": lhand_layer,
            "rhand_layer": rhand_layer,
            "dataset_name": config.dataset.name,
            "is_lhand": is_lhand,
            "is_rhand": is_rhand,
            "top_mask": top_mask,
            "weight": config.sampling.guidance.weight,
            "epsilon": config.sampling.guidance.epsilon,
            "threshold": config.sampling.guidance.threshold,
            "normalize_gradient": config.sampling.guidance.normalize_gradient,
            "start_ratio": config.sampling.guidance.start_ratio,
        }

    result = sample_ddim(
        diffusion,
        model,
        condition,
        text,
        nframes,
        hand_features=99,
        object_features=config.model.obj_nfeats,
        dmap_features=2048,
        valid_lhand=valid_lhand,
        valid_rhand=valid_rhand,
        valid_object=valid_object,
        steps=config.sampling.steps,
        eta=config.sampling.eta,
        cfg_scale=config.sampling.cfg_scale,
        seed=args.seed,
        cig=cig,
    )

    arrays = {name: tensor[0].detach().cpu().numpy() for name, tensor in result.items()}
    if not is_lhand:
        arrays["lhand"].fill(0.0)
    if not is_rhand:
        arrays["rhand"].fill(0.0)
    split = arrays["dmap"].shape[-1] // 2
    left = (arrays["dmap"][:, :split] + 1.0) * 0.5
    right = (arrays["dmap"][:, split:] + 1.0) * 0.5
    arrays["dmap_meters"] = np.concatenate(
        [
            left * (stats["ldmap_max"] - stats["ldmap_min"]) + stats["ldmap_min"],
            right * (stats["rdmap_max"] - stats["rdmap_min"]) + stats["rdmap_min"],
        ],
        axis=-1,
    ).clip(min=0.0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **arrays,
        prompt=args.prompt,
        object_name=args.object_name,
        is_lhand=is_lhand,
        is_rhand=is_rhand,
        nframes=nframes,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
