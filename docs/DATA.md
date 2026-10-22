# Processed data contract

JointHOI uses the public [GRAB](https://github.com/otaheri/GRAB) and [ARCTIC](https://github.com/zc-alexfan/arctic) datasets. Data generation, preprocessing, and third-party dataset files are intentionally excluded. Training expects one NumPy archive per dataset plus the object/text metadata used by [Text2HOI](https://github.com/JunukCha/Text2HOI).

## `data.npz`

Required arrays for both datasets:

| Key | Description |
| --- | --- |
| `x_lhand` | Left MANO motion, `[N, T, 99]` or an object array of `[T, 99]` sequences |
| `x_rhand` | Right MANO motion, same layout |
| `x_obj` | Object translation and 6D rotation, `[N, T, 9]` |
| `is_lhand`, `is_rhand` | Per-sequence hand-presence flags |
| `action_name` | Per-sequence action labels |
| `nframes` | Original sequence lengths |
| `lhand_min_dist_raw` | Object-anchor-to-left-hand distances in meters, `[N, T, 1024]` |
| `rhand_min_dist_raw` | Object-anchor-to-right-hand distances in meters, `[N, T, 1024]` |
| `object_name` or `obj_name` | Object key used by `obj.pkl` |

ARCTIC additionally requires `x_obj_angle` with shape `[N, T, 1]`. GRAB may include `proc_obj_name`, which is used only to find text descriptions.

## `dmap_stats.npz`

The four scalar keys are `ldmap_min`, `ldmap_max`, `rdmap_min`, and `rdmap_max`. Training maps each hand's raw distances to `[-1, 1]` independently and clips outliers to that interval.

The public configurations use the front-aligned crop (`random_start: false`). If random cropping is enabled, provide statistics computed with the same protocol and change `stats_path` accordingly.

## `obj.pkl`

Expected keys are `object_name`, `obj_pcs`, `obj_pc_normals`, `point_sets`, and optionally `obj_path` and `obj_pc_top`. Each point cloud must contain the same 1024 anchors used to compute the distance maps. `obj_pc_top` is needed for articulated ARCTIC objects.

## `text.json`

Maps canonical prompt keys to lists of text descriptions. If a key is missing, the loader falls back to a deterministic template.
