import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from jointhoi.data import ProcessedHOIDataset, build_dataloader
from jointhoi.diffusion import GaussianDiffusion
from jointhoi.models import JointHOI, PointNetfeat
from jointhoi.models.text import encode_text, load_clip


def parse_args():
    parser = argparse.ArgumentParser(description="Train JointHOI")
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument(
        "overrides", nargs="*", help="OmegaConf dot-list overrides, e.g. train.batch_size=16"
    )
    return parser.parse_args()


def load_config(path, overrides):
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    return config


def build_condition(pointnet, points, scale, centroid):
    with torch.no_grad():
        features = pointnet(points)
        if features.ndim == 3:
            features = features[:, 0]
    if scale.ndim == 1:
        scale = scale[:, None]
    return torch.cat([features, scale, centroid], dim=-1)


def make_scheduler(optimizer, warmup_steps, total_steps):
    def multiplier(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def load_pointnet(config, device):
    model = PointNetfeat(
        global_feat=False,
        feature_transform=False,
        in_dim=3,
    ).to(device)
    checkpoint = torch.load(config.pointnet_checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval().requires_grad_(False)
    return model


def save_checkpoint(path, model, optimizer, scheduler, step, best_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        },
        path,
    )


def main():
    args = parse_args()
    config = load_config(args.config, args.overrides)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    dataset = ProcessedHOIDataset(**OmegaConf.to_container(config.dataset, resolve=True))
    dataloader = build_dataloader(
        dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        balance_weights_path=config.dataset.get("balance_weights_path"),
    )
    pointnet = load_pointnet(config, device)
    clip_model = load_clip(config.clip_version, device)

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
    diffusion = GaussianDiffusion(
        steps=config.diffusion.steps,
        schedule=config.diffusion.schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    scheduler = make_scheduler(optimizer, config.train.warmup_steps, config.train.max_steps)
    start_step = 0
    best_loss = float("inf")
    if config.train.resume:
        checkpoint = torch.load(config.train.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_step = int(checkpoint.get("global_step", 0))
        best_loss = float(checkpoint.get("best_loss", best_loss))

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml")
    weights = OmegaConf.to_container(config.train.loss_weights, resolve=True)
    iterator = iter(dataloader)
    running = 0.0
    model.train()

    for step in range(start_step, config.train.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        tensors = {
            key: batch[key].to(device, non_blocking=True)
            for key in (
                "x_lhand", "x_rhand", "x_obj", "x_dmap", "normalized_obj_pc",
                "obj_scale", "obj_cent", "valid_mask_lhand", "valid_mask_rhand",
                "valid_mask_obj",
            )
        }
        condition = build_condition(
            pointnet,
            tensors["normalized_obj_pc"],
            tensors["obj_scale"],
            tensors["obj_cent"],
        )
        text = encode_text(clip_model, list(batch["text"]), device)

        keep_dmap = torch.rand(len(text), device=device) >= config.train.dmap_dropout
        x_dmap = tensors["x_dmap"].clone()
        if (~keep_dmap).any():
            x_dmap[~keep_dmap] = model.null_dmap.expand(
                int((~keep_dmap).sum()), x_dmap.shape[1], x_dmap.shape[2]
            )

        losses = diffusion.training_losses(
            model,
            tensors["x_lhand"],
            tensors["x_rhand"],
            tensors["x_obj"],
            x_dmap,
            condition,
            text,
            tensors["valid_mask_lhand"],
            tensors["valid_mask_rhand"],
            tensors["valid_mask_obj"],
            keep_dmap,
        )
        loss = sum(float(weights[name]) * value for name, value in losses.items())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
        optimizer.step()
        scheduler.step()

        current_step = step + 1
        running += float(loss.detach())
        if current_step % config.train.log_every == 0:
            detail = " ".join(f"{name}={float(value):.4f}" for name, value in losses.items())
            print(
                f"step={current_step}/{config.train.max_steps} "
                f"loss={running / config.train.log_every:.4f} {detail}"
            )
            running = 0.0

        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            torch.save(
                {"global_step": current_step, "model": model.state_dict(), "best_loss": best_loss},
                output / "best.pth",
            )
        if current_step % config.train.save_every == 0:
            save_checkpoint(
                output / f"step_{current_step}.pth",
                model,
                optimizer,
                scheduler,
                current_step,
                best_loss,
            )

    save_checkpoint(
        output / "last.pth", model, optimizer, scheduler, config.train.max_steps, best_loss
    )


if __name__ == "__main__":
    main()
