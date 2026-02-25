import cv2
import yaml
import numpy as np
from typing import Tuple


class CameraCalibration:
    """Camera intrinsic + extrinsic calibration utilities."""

    def __init__(self):
        self.mtx: np.ndarray | None = None
        self.dist: np.ndarray | None = None
        self.rvec: np.ndarray | None = None
        self.tvec: np.ndarray | None = None

    # ==============================================================
    # Intrinsic
    # ==============================================================
    def load_intrinsic(self, yaml_file: str) -> None:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)["CameraInt"]

        self.mtx = np.array([
            [data["fx"], 0, data["cx"]],
            [0, data["fy"], data["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist = np.array(data["dist_coeffs"], dtype=np.float32).reshape(1, -1)

    def load_extrinsic(self, yaml_file: str) -> None:
        with open(yaml_file, "r") as f:
            ext = yaml.safe_load(f)["CameraExt"]

        self.rvec = np.array(ext["rvec"], dtype=np.float32).reshape(3, 1)
        self.tvec = np.array(ext["tvec"], dtype=np.float32).reshape(3, 1)

    # ==============================================================
    # Projection
    # ==============================================================
    def pixel_to_world(self, u: float, v: float, z_world: float = -730.0) -> Tuple[float, float, float]:
        """Back-project pixel to world assuming Z = constant plane."""
        if self.mtx is None or self.dist is None or self.rvec is None:
            raise RuntimeError("Camera not calibrated")

        pts = np.array([[[u, v]]], dtype=np.float32)
        norm = cv2.undistortPoints(pts, self.mtx, self.dist)[0, 0]

        ray_cam = np.array([[norm[0]], [norm[1]], [1.0]], dtype=np.float32)

        R, _ = cv2.Rodrigues(self.rvec)
        R_inv = R.T
        cam_pos = -R_inv @ self.tvec
        ray_world = R_inv @ ray_cam

        if abs(ray_world[2, 0]) < 1e-6:
            raise ValueError("Ray parallel to plane")

        s = (z_world - cam_pos[2, 0]) / ray_world[2, 0]
        Pw = cam_pos + s * ray_world
        return tuple(float(x) for x in Pw.flatten())
