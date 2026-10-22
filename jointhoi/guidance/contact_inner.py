import torch

from jointhoi.utils.rotation import axis_angle_to_quaternion, quaternion_apply, rot6d_to_axis_angle, rot6d_to_rotmat


def _hand_vertices(parameters, mano_layer):
    frames = parameters.shape[0]
    translation = parameters[:, :3]
    pose = rot6d_to_axis_angle(parameters[:, 3:]).reshape(frames, 48)
    output = mano_layer(
        betas=torch.zeros((frames, 10), device=parameters.device),
        global_orient=pose[:, :3],
        hand_pose=pose[:, 3:],
    )
    return output.vertices + translation[:, None]


def transform_object(points, parameters, dataset_name, top_mask=None):
    frames = parameters.shape[0]
    rotation = rot6d_to_rotmat(parameters[:, 3:9]).reshape(frames, 3, 3)
    points = points[:1024]

    if dataset_name == "arctic" and parameters.shape[-1] == 10 and top_mask is not None:
        expanded = points[None].expand(frames, -1, -1).clone()
        top_mask = top_mask.bool()
        angle = parameters[:, 9:10]
        axis = torch.tensor([0.0, 0.0, -1.0], device=parameters.device)
        quaternion = axis_angle_to_quaternion(axis[None] * angle)
        expanded[:, top_mask] = quaternion_apply(
            quaternion[:, None], expanded
        )[:, top_mask]
        rotated = torch.einsum("tij,tkj->tki", rotation, expanded)
    elif dataset_name == "grab":
        rotated = torch.einsum("tij,ki->tkj", rotation, points)
    else:
        rotated = torch.einsum("tij,kj->tki", rotation, points)
    return rotated + parameters[:, None, :3]


def distance_map(hand_vertices, object_points):
    return torch.cdist(object_points, hand_vertices).amin(dim=-1)


def log_l1_contact_energy(implied, generated, epsilon=5e-3, threshold=None):
    """Equation (15): L1 distance between log-scaled distance maps."""
    if threshold is not None and threshold < 1.0:
        mask = generated < threshold
        if not mask.any():
            return implied.sum() * 0.0
        implied = implied[mask]
        generated = generated[mask]
    return (torch.log(implied + epsilon) - torch.log(generated + epsilon)).abs().mean()


def apply_contact_inner_guidance(
    pred_lhand,
    pred_rhand,
    pred_obj,
    pred_dmap,
    object_points,
    stats,
    lhand_layer,
    rhand_layer,
    dataset_name,
    is_lhand=True,
    is_rhand=True,
    top_mask=None,
    weight=0.05,
    epsilon=5e-3,
    threshold=None,
    normalize_gradient=True,
):
    """Apply the release implementation of Contact Inner Guidance.

    The research sampler updates the current clean hand estimate, then performs
    the DDIM reverse step. Batch size one is used by the public prompt sampler.
    """
    if pred_lhand.shape[0] != 1:
        raise ValueError("Contact Inner Guidance currently expects batch size 1")

    generated = pred_dmap[0].detach()
    split = generated.shape[-1] // 2
    left = (generated[:, :split] + 1.0) * 0.5
    right = (generated[:, split:] + 1.0) * 0.5
    left = (left * (stats["ldmap_max"] - stats["ldmap_min"]) + stats["ldmap_min"]).clamp_min(0.0)
    right = (right * (stats["rdmap_max"] - stats["rdmap_min"]) + stats["rdmap_min"]).clamp_min(0.0)

    transformed_object = transform_object(
        object_points, pred_obj[0].detach(), dataset_name, top_mask
    ).detach()

    adjusted_left = pred_lhand
    adjusted_right = pred_rhand
    diagnostics = {}
    for side, enabled, prediction, target, layer in (
        ("lhand", is_lhand, pred_lhand, left, lhand_layer),
        ("rhand", is_rhand, pred_rhand, right, rhand_layer),
    ):
        if not enabled:
            continue
        with torch.enable_grad():
            parameters = prediction.detach().clone().requires_grad_(True)
            implied = distance_map(_hand_vertices(parameters[0], layer), transformed_object)
            energy = log_l1_contact_energy(implied, target, epsilon, threshold)
            gradient = torch.autograd.grad(energy, parameters)[0]
            if normalize_gradient:
                gradient = gradient / gradient.norm().clamp_min(1e-8)
            updated = prediction - weight * gradient
        diagnostics[side] = float(energy.detach())
        if side == "lhand":
            adjusted_left = updated
        else:
            adjusted_right = updated
    return adjusted_left, adjusted_right, diagnostics
