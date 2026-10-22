import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .objects import ObjectRepository


def normalize_point_cloud(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    scale = np.linalg.norm(centered, axis=1).max()
    if scale <= 0:
        raise ValueError("Object point cloud has zero spatial extent")
    return centered / scale, centroid.astype(np.float32), np.float32(scale)


def _pad(sequence, length, width):
    result = np.zeros((length, width), dtype=np.float32)
    sequence = np.asarray(sequence, dtype=np.float32)
    count = min(length, len(sequence))
    result[:count] = sequence[:count]
    return result


def _fallback_prompt(action, object_name, is_lhand, is_rhand):
    if is_lhand and is_rhand:
        hand_phrase = "both hands"
    elif is_lhand:
        hand_phrase = "left hand"
    else:
        hand_phrase = "right hand"
    return f"{action} {object_name} with {hand_phrase}."


def _normalize_grab_object_name(name):
    name = re.sub(r"(large|medium|small|middle)$", "", name.lower().strip()).strip()
    aliases = {
        "alarmclock": "alarm clock",
        "waterbottle": "water bottle",
        "gamecontroller": "game controller",
        "piggybank": "piggy bank",
        "stanfordbunny": "stanford bunny",
        "lightbulb": "light bulb",
        "fryingpan": "frying pan",
        "coffeemug": "mug",
        "wristwatch": "watch",
        "rubberduck": "duck",
    }
    return aliases.get(name, name)


class ProcessedHOIDataset(Dataset):
    """Loader for preprocessed GRAB/ARCTIC JointHOI arrays.

    Preprocessing itself is intentionally outside this release. The expected
    archive schema is documented in ``docs/DATA.md``.
    """

    def __init__(
        self,
        name,
        data_path,
        object_path,
        text_path,
        stats_path,
        max_nframes,
        random_start=False,
        balance_weights_path=None,
        flat_hand=False,
    ):
        self.name = name.lower()
        if self.name not in {"grab", "arctic"}:
            raise ValueError("name must be 'grab' or 'arctic'")
        self.max_nframes = int(max_nframes)
        self.random_start = bool(random_start)
        self.flat_hand = bool(flat_hand)
        self.object_repository = ObjectRepository(object_path)

        with np.load(data_path, allow_pickle=True) as archive:
            self.data = {key: archive[key] for key in archive.files}

        with Path(text_path).open("r") as handle:
            self.text_descriptions = json.load(handle)

        with np.load(stats_path) as stats:
            self.ldmap_min = float(stats["ldmap_min"])
            self.ldmap_max = float(stats["ldmap_max"])
            self.rdmap_min = float(stats["rdmap_min"])
            self.rdmap_max = float(stats["rdmap_max"])

        self.object_names = self.data.get("object_name", self.data.get("obj_name"))
        if self.object_names is None:
            raise KeyError("Processed archive needs 'object_name' or 'obj_name'")
        self.processed_object_names = self.data.get("proc_obj_name", self.object_names)

        required = {
            "x_lhand", "x_rhand", "x_obj", "is_lhand", "is_rhand",
            "action_name", "nframes", "lhand_min_dist_raw", "rhand_min_dist_raw",
        }
        missing = sorted(required.difference(self.data))
        if missing:
            raise KeyError(f"Processed archive is missing: {', '.join(missing)}")

    def __len__(self):
        return len(self.data["nframes"])

    def inverse_distance_map(self, normalized):
        split = normalized.shape[-1] // 2
        left = (normalized[..., :split] + 1.0) * 0.5
        right = (normalized[..., split:] + 1.0) * 0.5
        left = left * (self.ldmap_max - self.ldmap_min) + self.ldmap_min
        right = right * (self.rdmap_max - self.rdmap_min) + self.rdmap_min
        if torch.is_tensor(normalized):
            return torch.cat([left, right], dim=-1).clamp_min(0.0)
        return np.clip(np.concatenate([left, right], axis=-1), 0.0, None)

    def _prompt(self, index, action, object_name, is_lhand, is_rhand):
        prompt_object = object_name
        if self.name == "grab":
            prompt_object = _normalize_grab_object_name(str(self.processed_object_names[index]))
        prompt = _fallback_prompt(action, prompt_object, is_lhand, is_rhand)
        key = prompt.capitalize()
        descriptions = self.text_descriptions.get(key)
        if descriptions is None and self.name == "grab":
            key = _fallback_prompt(action, object_name, is_lhand, is_rhand).capitalize()
            descriptions = self.text_descriptions.get(key)
        return str(np.random.choice(descriptions)) if descriptions else prompt

    def __getitem__(self, index):
        sequence_length = int(self.data["nframes"][index])
        is_lhand = bool(self.data["is_lhand"][index])
        is_rhand = bool(self.data["is_rhand"][index])
        used_length = min(sequence_length, self.max_nframes)
        if self.random_start and sequence_length > self.max_nframes:
            start = np.random.randint(0, sequence_length - self.max_nframes + 1)
        else:
            start = 0

        x_lhand = _pad(self.data["x_lhand"][index][start:], self.max_nframes, 99)
        x_rhand = _pad(self.data["x_rhand"][index][start:], self.max_nframes, 99)
        x_obj_base = _pad(self.data["x_obj"][index][start:], self.max_nframes, 9)
        if self.name == "arctic":
            if "x_obj_angle" not in self.data:
                raise KeyError("ARCTIC archive needs 'x_obj_angle'")
            angle = _pad(self.data["x_obj_angle"][index][start:], self.max_nframes, 1)
            x_obj = np.concatenate([x_obj_base, angle], axis=-1)
        else:
            x_obj = x_obj_base

        if not is_lhand:
            x_lhand.fill(0.0)
        if not is_rhand:
            x_rhand.fill(0.0)

        left_raw = _pad(
            self.data["lhand_min_dist_raw"][index][start:], self.max_nframes, 1024
        )
        right_raw = _pad(
            self.data["rhand_min_dist_raw"][index][start:], self.max_nframes, 1024
        )
        left_range = max(self.ldmap_max - self.ldmap_min, 1e-8)
        right_range = max(self.rdmap_max - self.rdmap_min, 1e-8)
        left = np.clip(2.0 * (left_raw - self.ldmap_min) / left_range - 1.0, -1.0, 1.0)
        right = np.clip(2.0 * (right_raw - self.rdmap_min) / right_range - 1.0, -1.0, 1.0)
        if not is_lhand:
            left.fill(0.0)
        if not is_rhand:
            right.fill(0.0)

        object_name = str(self.object_names[index])
        obj = self.object_repository.get(object_name)
        normalized_pc, centroid, scale = normalize_point_cloud(obj["points"])
        valid = np.arange(self.max_nframes) < used_length

        action = str(self.data["action_name"][index])
        item = {
            "x_lhand": x_lhand,
            "x_rhand": x_rhand,
            "x_obj": x_obj,
            "x_dmap": np.concatenate([left, right], axis=-1).astype(np.float32),
            "normalized_obj_pc": normalized_pc.astype(np.float32),
            "obj_cent": centroid,
            "obj_scale": scale,
            "valid_mask_lhand": valid & is_lhand,
            "valid_mask_rhand": valid & is_rhand,
            "valid_mask_obj": valid,
            "text": self._prompt(index, action, object_name, is_lhand, is_rhand),
            "object_name": object_name,
            "is_lhand": is_lhand,
            "is_rhand": is_rhand,
        }
        return item


def build_dataloader(dataset, batch_size, num_workers, balance_weights_path=None):
    sampler = None
    shuffle = balance_weights_path is None
    if balance_weights_path is not None:
        with Path(balance_weights_path).open("rb") as handle:
            weights = pickle.load(handle)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.float32), len(dataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
