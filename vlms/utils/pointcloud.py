from __future__ import annotations

import numpy as np
import open3d as o3d


def depth_to_point_cloud(
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    intrinsic_matrix: np.ndarray,
    clip_features: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Convert a depth image to a point cloud.

    Args:
        depth_image: Depth image of shape (H, W), float32 in metres.
        rgb_image: RGB image of shape (H, W, 3), uint8.
        intrinsic_matrix: Camera intrinsic matrix of shape (3, 3).
        clip_features: Optional CLIP features of shape (H, W, C).

    Returns:
        xyz:  (N, 3) float32 camera-space coordinates.
        rgb:  (N, 3) uint8 colours.
        feats: (N, C) float32 features, or None if clip_features was None.
    """
    assert depth_image.dtype == np.float32, "Depth image should be of type float32"
    assert (
        rgb_image.dtype == np.uint8 and rgb_image.shape[:2] == depth_image.shape
    ), "RGB image should be of type uint8"
    assert intrinsic_matrix.shape == (3, 3), "Intrinsic matrix should be of shape (3, 3)"

    h, w = depth_image.shape
    y_indices, x_indices = np.mgrid[0:h, 0:w]

    z_flat = depth_image.ravel()
    valid_mask = z_flat > 0

    x_flat = x_indices.ravel()[valid_mask].astype(np.float32)
    y_flat = y_indices.ravel()[valid_mask].astype(np.float32)
    z_flat = z_flat[valid_mask]

    x_camera = (x_flat - intrinsic_matrix[0, 2]) * z_flat / intrinsic_matrix[0, 0]
    y_camera = (y_flat - intrinsic_matrix[1, 2]) * z_flat / intrinsic_matrix[1, 1]

    xyz = np.stack((x_camera, y_camera, z_flat), axis=-1)  # (N, 3)
    rgb = rgb_image.reshape(-1, 3)[valid_mask]               # (N, 3)

    feats = None
    if clip_features is not None:
        feats = clip_features.reshape(-1, clip_features.shape[-1])[valid_mask]  # (N, C)

    return xyz, rgb, feats


def compute_world_bounds_from_depth(
    depth_image: np.ndarray,
    intrinsic_matrix: np.ndarray,
    camera_pose: np.ndarray,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-space min/max bounds from a depth image using strided sampling.

    Args:
        depth_image: (H, W) float32 depth in meters.
        intrinsic_matrix: (3, 3) camera intrinsic matrix.
        camera_pose: (4, 4) cam-to-world transformation.
        stride: Subsample every ``stride``-th pixel in both dimensions.

    Returns:
        (min_xyz, max_xyz): each shape (3,).
    """
    h, w = depth_image.shape
    y_indices, x_indices = np.mgrid[0:h:stride, 0:w:stride]
    x_flat = x_indices.ravel().astype(np.float32)
    y_flat = y_indices.ravel().astype(np.float32)
    z_flat = depth_image[0:h:stride, 0:w:stride].ravel()

    valid = z_flat > 0
    x_flat, y_flat, z_flat = x_flat[valid], y_flat[valid], z_flat[valid]

    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    x_cam = (x_flat - cx) * z_flat / fx
    y_cam = (y_flat - cy) * z_flat / fy
    pts_cam = np.stack([x_cam, y_cam, z_flat], axis=1)  # (N, 3)

    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    pts_world = (R @ pts_cam.T).T + t  # (N, 3)

    return pts_world.min(axis=0), pts_world.max(axis=0)


def create_posed_cloud(
    depth_image: np.ndarray,
    color_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    clip_features: np.ndarray = None,
    with_visualization: bool = False,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.LineSet]:
    """
    Create a posed point cloud from depth and color images.

    Args:
        depth_image (np.ndarray): Depth image of shape (H, W).
        color_image (np.ndarray): Color image of shape (H, W, 3).
        camera_intrinsics (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        camera_pose (np.ndarray): Camera pose matrix of shape (4, 4). Expect cam to world transformation.
        with_visualization (bool): Whether to create a camera visualization.

    Returns:
        tuple: A tuple containing:
            - o3d.geometry.PointCloud: The posed point cloud.
            - o3d.geometry.LineSet: The camera visualization lines, if requested.
    """
    xyz, rgb, _feats = depth_to_point_cloud(
        depth_image, color_image, camera_intrinsics, clip_features
    )

    # Apply camera pose transformation
    pts_world = (camera_pose[:3, :3] @ xyz.T).T + camera_pose[:3, 3]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts_world)
    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

    camera_lines = None
    if with_visualization:
        # camera visualization wants world to camera transformation
        camera_pose_inv = np.linalg.inv(camera_pose)
        # Create a line set for the camera pose
        camera_lines = o3d.geometry.LineSet.create_camera_visualization(
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=color_image.shape[1],
                height=color_image.shape[0],
                fx=camera_intrinsics[0, 0],
                fy=camera_intrinsics[1, 1],
                cx=camera_intrinsics[0, 2],
                cy=camera_intrinsics[1, 2],
            ),
            extrinsic=camera_pose_inv,
            scale=0.1,  # Scale for the camera visualization
        )

    return pc, camera_lines


def scale_intrinsics(
    K: np.ndarray,
    target_hw: tuple[int, int],
    orig_hw: tuple[int, int],
) -> np.ndarray:
    """Scale a camera intrinsic matrix to a new image resolution.

    Args:
        K: (3, 3) or (3, 4) camera matrix (modified copy returned).
        target_hw: (height, width) of the target image.
        orig_hw: (height, width) of the original image.

    Returns:
        Scaled copy of K.
    """
    scale_x = target_hw[1] / orig_hw[1]
    scale_y = target_hw[0] / orig_hw[0]
    K_out = K.copy()
    K_out[0, 0] *= scale_x  # fx
    K_out[1, 1] *= scale_y  # fy
    K_out[0, 2] *= scale_x  # cx
    K_out[1, 2] *= scale_y  # cy
    return K_out


def build_cam_pose(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Build a 4x4 cam-to-world matrix from a rotation matrix and translation.

    Args:
        R: (3, 3) rotation matrix.
        T: (3,) translation vector.

    Returns:
        (4, 4) float64 homogeneous transformation matrix.
    """
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = T
    return pose


class SparseVoxelGrid:
    """Surface-conforming sparse voxel grid for fusing per-point features.

    Only allocates storage for voxels that contain at least one point, so
    memory scales with the occupied surface area instead of the full bounding
    volume.  Each voxel tracks the centroid of its contributing points
    (not the cell centre), producing positions that closely follow the
    underlying geometry.
    """

    def __init__(self, voxel_size: float, origin: np.ndarray | None = None):
        self.voxel_size = voxel_size
        self.origin = origin
        self._voxels: dict[tuple[int, ...], dict] = {}

    def integrate(self, points_world: np.ndarray, features: np.ndarray) -> None:
        """Fuse a batch of world-space points and their features.

        Args:
            points_world: (N, 3) float array of world coordinates.
            features: (N, C) float array of per-point feature vectors.
        """
        if len(points_world) == 0:
            return
        if self.origin is None:
            self.origin = points_world.min(axis=0) - self.voxel_size

        ijk = np.floor((points_world - self.origin) / self.voxel_size).astype(np.int64)

        unique, inverse, counts = np.unique(
            ijk, axis=0, return_inverse=True, return_counts=True
        )

        n = len(unique)
        feat_sum = np.zeros((n, features.shape[1]), dtype=np.float64)
        pos_sum = np.zeros((n, 3), dtype=np.float64)
        np.add.at(feat_sum, inverse, features.astype(np.float64))
        np.add.at(pos_sum, inverse, points_world.astype(np.float64))

        for i in range(n):
            key = tuple(unique[i])
            c = int(counts[i])
            avg_feat = feat_sum[i] / c
            avg_pos = pos_sum[i] / c

            if key in self._voxels:
                v = self._voxels[key]
                old = v["count"]
                total = old + c
                v["features"] = (v["features"] * old + avg_feat * c) / total
                v["centroid"] = (v["centroid"] * old + avg_pos * c) / total
                v["count"] = total
            else:
                self._voxels[key] = {
                    "features": avg_feat,
                    "centroid": avg_pos,
                    "count": c,
                }

    def get_centroids(self) -> np.ndarray:
        """(M, 3) centroids of occupied voxels."""
        if not self._voxels:
            return np.empty((0, 3), dtype=np.float64)
        return np.array([v["centroid"] for v in self._voxels.values()])

    def get_features(self) -> np.ndarray:
        """(M, C) averaged feature vectors."""
        if not self._voxels:
            return np.empty((0, 0), dtype=np.float64)
        return np.array([v["features"] for v in self._voxels.values()])

    def get_counts(self) -> np.ndarray:
        """(M,) observation count per voxel."""
        if not self._voxels:
            return np.empty(0, dtype=np.int64)
        return np.array([v["count"] for v in self._voxels.values()])

    def __len__(self) -> int:
        return len(self._voxels)
