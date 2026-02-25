
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from camera_calibration import CameraCalibration
from typing import Dict, List, Tuple, Optional


class Map:
    """Unified world-map for multi-camera projection and merging."""

    def __init__(self, map_image_path: str, mm_per_pixel_x: float, mm_per_pixel_y: float):
        self.map_image_path = map_image_path
        self.mm_per_pixel_x = mm_per_pixel_x
        self.mm_per_pixel_y = mm_per_pixel_y

        self.map_img_original: np.ndarray | None = None
        self.map_img: np.ndarray | None = None  
        self.cameras: Dict[str, CameraCalibration] = {}
        self.camera_origins: Dict[str, Tuple[int, int]] = {}
        self.projected: List[dict] = []

        self.rois: List[dict] = []
        self.roi_mask: Optional[np.ndarray] = None
        self.roi_id_map: Dict[int, dict] = {}
    # ==============================================================
    # Setup
    # ==============================================================
    def _load_map(self) -> None:
        if self.map_img_original is None:
            self.map_img_original = cv2.imread(self.map_image_path)
            if self.map_img_original is None:
                raise FileNotFoundError(self.map_image_path)

    def add_camera(self, camera_id: str, intrinsic: str, extrinsic: str, origin_px: Optional[Tuple[int, int]] = None):
        cam = CameraCalibration()
        cam.load_intrinsic(intrinsic)
        cam.load_extrinsic(extrinsic)

        self.cameras[camera_id] = cam
        
        if origin_px is not None:
            self.camera_origins[camera_id] = origin_px
        else:
            # Load origin_px from extrinsic yaml
            with open(extrinsic, "r") as f:
                ext_data = yaml.safe_load(f).get("CameraExt", {})
                if "origin_px" in ext_data:
                    self.camera_origins[camera_id] = tuple(ext_data["origin_px"])
                else:
                    print(f"Warning: origin_px not found in {extrinsic} for {camera_id}")
                    self.camera_origins[camera_id] = (0, 0)

    def load_rois_from_yaml(self, roi_yaml_path):
        with open(roi_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "rois" not in data:
            raise ValueError("YAML does not contain 'rois' key")

        self.rois = []
        self.roi_id_map = {}

        for idx, roi in enumerate(data["rois"]):
            points = np.array(roi["points_px"], dtype=np.float32)
            
            # Auto-generate ID if not present (F2/F4 format uses 'name' only)
            roi_id = int(roi["id"]) if "id" in roi else (idx + 1)
            roi_label = roi.get("label") or roi.get("name") or f"ROI {roi_id}"
            
            # Handle rotation if angle is specified
            if "angle" in roi and roi["angle"] != 0:
                angle = roi["angle"]
                
                # Calculate center of the bounding box
                center_x = np.mean(points[:, 0])
                center_y = np.mean(points[:, 1])
                center = np.array([center_x, center_y])
                
                # Create rotation matrix
                angle_rad = np.radians(angle)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                rotation_matrix = np.array([
                    [cos_a, -sin_a],
                    [sin_a, cos_a]
                ])
                
                # Rotate points around center
                rotated_points = []
                for point in points:
                    # Translate to origin
                    translated = point - center
                    # Rotate
                    rotated = rotation_matrix @ translated
                    # Translate back
                    final_point = rotated + center
                    rotated_points.append(final_point)
                
                contour = np.array(rotated_points, dtype=np.int32)
            else:
                contour = points.astype(np.int32)

            roi_obj = {
                "id": roi_id,
                "label": roi_label,
                "points_px": roi["points_px"],
                "angle": roi.get("angle", 0),
                "contour": contour
            }

            self.rois.append(roi_obj)
            self.roi_id_map[roi_obj["id"]] = roi_obj
        self._build_roi_mask()
        print(f"✓ Loaded {len(self.rois)} ROIs")
    
    def _build_roi_mask(self):
        self._load_map()
        h, w = self.map_img_original.shape[:2]

        self.roi_mask = np.zeros((h, w), dtype=np.int16)

        for roi in self.rois:
            cv2.fillPoly(self.roi_mask, [roi["contour"]], roi["id"])

    # ==============================================================
    # Coordinate transform
    # ==============================================================
    def world_to_map(self, camera_id: str, X: float, Y: float) -> Tuple[int, int]:
        ox, oy = self.camera_origins[camera_id]
        u = int(ox + X / self.mm_per_pixel_x)
        v = int(oy + Y / self.mm_per_pixel_y)
        return u, v

    def map_to_world(self, camera_id: str, u: int, v: int) -> Tuple[float, float]:
        ox, oy = self.camera_origins[camera_id]
        X = (u - ox) * self.mm_per_pixel_x
        Y = (v - oy) * self.mm_per_pixel_y
        return X, Y

    # ==============================================================
    # Projection
    # ==============================================================
    def image_to_map(self, camera_points: Dict[str, List[dict]], z_world=-850.0):
        for cam_id, pts in camera_points.items():
            calib = self.cameras[cam_id]
            for p in pts:
                # Support both old tuple (pid, u, v) and new dict format
                if isinstance(p, (tuple, list)):
                    pid, u, v = p
                    face_name = "Unknown"
                else:
                    pid = p["id"]
                    u = p["u"]
                    v = p["v"]
                    face_name = p.get("face_name", "Unknown")

                wx, wy, wz = calib.pixel_to_world(u, v, z_world)
                mu, mv = self.world_to_map(cam_id, wx, wy)

                self.projected.append({
                    "camera": cam_id,
                    "point_id": pid,
                    "map_px": (mu, mv),
                    "world_mm": (wx, wy, wz),
                    "face_name": face_name
                })

    # ==============================================================
    # ROI
    # ==============================================================
    # def find_point_in_roi(self, merged_points: List[dict]) -> dict:
    #     """
    #     Returns a dict: {roi_id: point_id}
    #     """
    #     roi_has_point = {}

    #     for p in merged_points:
    #         u, v = p["map_px"]
    #         pid = p["point_id"]

    #         for roi in self.rois:
    #             if cv2.pointPolygonTest(roi["contour"], (u, v), False) >= 0:
    #                 roi_has_point[roi["id"]] = pid
    #                 break

    #     return roi_has_point


    def find_points_in_roi(self, points: List[dict]) -> Dict[int, bool]:
        """
        Return a map: {roi_id: True if contains at least one point}
        """
        if self.roi_mask is None:
            return {}

        h, w = self.roi_mask.shape

        roi_has_points: Dict[int, bool] = {
            roi["id"]: False for roi in self.rois
        }

        for p in points:
            u, v = p["map_px"]

            if 0 <= u < w and 0 <= v < h:
                roi_id = int(self.roi_mask[v, u])
                if roi_id != 0:
                    roi_has_points[roi_id] = True

        return roi_has_points

    # ==============================================================
    # Merge
    # ==============================================================
    def merge_points(self, distance_mm: float = 150, reference_camera:Optional[str] = None) -> List[dict]:
        if not self.projected:
            return []

        # mm → pixel threshold
        thresh_px = np.sqrt(
            (distance_mm / self.mm_per_pixel_x) ** 2 +
            (distance_mm / self.mm_per_pixel_y) ** 2
        )

        clusters = []

        for p in self.projected:
            pu, pv = p["map_px"]
            pid = p["point_id"]          #  numeric ID

            added = False
            for c in clusters:
                # DO NOT merge if ID is different
                if c["point_id"] != pid:
                    continue

                cu, cv = c["center"]
                if np.hypot(pu - cu, pv - cv) < thresh_px:
                    c["points"].append(p)

                    # update cluster center
                    us = [pt["map_px"][0] for pt in c["points"]]
                    vs = [pt["map_px"][1] for pt in c["points"]]
                    c["center"] = (np.mean(us), np.mean(vs))

                    added = True
                    break

            if not added:
                clusters.append({
                    "point_id": pid,      # store ID in cluster
                    "center": (pu, pv),
                    "points": [p]
                })

        if reference_camera is None:
            reference_camera = list(self.camera_origins.keys())[0]

        merged = []
        for c in clusters:
            cu, cv = c["center"]
            wx, wy = self.map_to_world(
                reference_camera,
                int(round(cu)),
                int(round(cv))
            )

            # Get metadata from first point in cluster (or aggregate)
            face_name = "Unknown"
            cameras = []
            for p in c["points"]:
                cameras.append(p["camera"])
                if p.get("face_name") and p["face_name"] != "Unknown":
                    face_name = p["face_name"]
            
            merged.append({
                "point_id": c["point_id"],
                "map_px": (int(round(cu)), int(round(cv))),
                "world_mm": (wx, wy),
                "face_name": face_name,
                "cameras": list(set(cameras))
            })

        return merged


    # ==============================================================
    # Drawing
    # ==============================================================
    # def draw_points_and_rois(self, merged_points: List[dict]) -> None:
    #     self._load_map()
        
    #     roi_has_points = self.find_point_in_roi(merged_points)

    #     for roi in self.rois:
    #         roi_id = roi["id"]
    #         pid = roi_has_points.get(roi_id)
    #         color = (0, 255, 0) if pid == roi_id else (0, 0, 255)
    #         cv2.polylines(self.map_img, [roi["contour"]], True, color, 2)

    #     for p in merged_points:
    #         u, v = p["map_px"]
    #         pid = p["point_id"]

    #         cv2.circle(self.map_img, (u, v), 4, (255, 255, 0), 2)
    #         cv2.putText(self.map_img, str(pid), (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
    def draw_points_and_rois(self, image:np.ndarray, merged_points: List[dict]) -> None:

        roi_has_points = self.find_points_in_roi(merged_points)

        for roi in self.rois:
            color = (0, 255, 0) if roi_has_points.get(roi["id"], False) else (0, 0, 255)
            cv2.polylines(image, [roi["contour"]], True, color, 2)

        for p in merged_points:
            u, v = p["map_px"]
            pid = p["point_id"]

            cv2.circle(image, (u, v), 4, (255, 255, 0), 2)
            cv2.putText(image, str(pid), (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
        return image

    def update_map(self, camera_points: Dict[str, List[Tuple[int, float, float]]], 
                distance_mm: float = 130, 
                reference_camera: Optional[str] = None) -> None:
      
        # Ensure map is loaded
        self._load_map()

        # Clear previous projections
        self.projected = []
        map_img = self.map_img_original.copy()
        # Project new camera points
        self.image_to_map(camera_points)

        # Merge points
        merged_points = self.merge_points(distance_mm=distance_mm, reference_camera=reference_camera)

        # Draw ROIs and merged points
        map_img = self.draw_points_and_rois(map_img, merged_points)

        return map_img
        
    def save_and_show(self, image:np.ndarray, path: str):
        
        cv2.imwrite(path, image)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
