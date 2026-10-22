import pickle
from pathlib import Path


class ObjectRepository:
    """Reader for the processed object point-cloud pickle used by Text2HOI."""

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as handle:
            data = pickle.load(handle)

        self.object_names = data["object_name"]
        self.point_clouds = data["obj_pcs"]
        self.normals = data["obj_pc_normals"]
        self.point_sets = data["point_sets"]
        self.mesh_paths = data.get("obj_path", {})
        self.top_masks = data.get("obj_pc_top")

    def get(self, object_name):
        if isinstance(object_name, int):
            object_name = self.object_names[object_name]
        if object_name not in self.point_clouds:
            choices = ", ".join(map(str, sorted(self.point_clouds)[:10]))
            raise KeyError(f"Unknown object '{object_name}'. Example available objects: {choices}")

        result = {
            "name": object_name,
            "point_set": self.point_sets[object_name].copy(),
            "points": self.point_clouds[object_name].copy(),
            "normals": self.normals[object_name].copy(),
            "mesh_path": self.mesh_paths.get(object_name),
            "top_mask": None,
        }
        if self.top_masks is not None:
            result["top_mask"] = self.top_masks[object_name].copy()
        return result
