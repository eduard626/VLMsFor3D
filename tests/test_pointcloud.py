"""Tests for the geometric pipeline: unprojection, intrinsic scaling, pose
construction, world transform, and sparse voxel fusion.

Each test uses synthetic data with known answers so failures pinpoint
exactly which stage is broken.
"""

import numpy as np
import numpy.testing as npt
import cv2
import pytest

from vlms.utils.pointcloud import (
    build_cam_pose,
    depth_to_point_cloud,
    scale_intrinsics,
    SparseVoxelGrid,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def _project(pt_cam: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    """Project a camera-space 3D point to pixel coordinates."""
    u = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
    v = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]
    return float(u), float(v)


def _rotation_y(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a), 0, np.sin(a)],
        [0, 1, 0],
        [-np.sin(a), 0, np.cos(a)],
    ])


# ── depth_to_point_cloud ────────────────────────────────────────────────────

class TestDepthToPointCloud:
    def test_center_pixel(self):
        """Pixel at the principal point unprojects along the optical axis."""
        depth = np.zeros((3, 3), dtype=np.float32)
        depth[1, 1] = 2.0
        rgb = np.zeros((3, 3, 3), dtype=np.uint8)
        rgb[1, 1] = [255, 128, 64]
        K = _make_K(100, 100, 1.0, 1.0)  # principal point = (1, 1)

        xyz, colors, _ = depth_to_point_cloud(depth, rgb, K)

        assert xyz.shape == (1, 3)
        npt.assert_allclose(xyz[0], [0.0, 0.0, 2.0], atol=1e-6)
        npt.assert_array_equal(colors[0], [255, 128, 64])

    def test_off_center_pixel(self):
        """Known off-center pixel produces the expected ray direction."""
        depth = np.zeros((3, 5), dtype=np.float32)
        depth[0, 4] = 3.0  # row=0, col=4
        rgb = np.zeros((3, 5, 3), dtype=np.uint8)
        K = _make_K(200, 200, 2.0, 1.0)

        xyz, _, _ = depth_to_point_cloud(depth, rgb, K)

        # x = (4 - 2) * 3 / 200 = 0.03
        # y = (0 - 1) * 3 / 200 = -0.015
        npt.assert_allclose(xyz[0], [0.03, -0.015, 3.0], atol=1e-6)

    def test_zero_depth_skipped(self):
        """Pixels with depth == 0 produce no output points."""
        depth = np.zeros((2, 2), dtype=np.float32)
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        K = _make_K(100, 100, 1, 1)

        xyz, colors, _ = depth_to_point_cloud(depth, rgb, K)

        assert xyz.shape == (0, 3)
        assert colors.shape == (0, 3)

    def test_features_passed_through(self):
        """Per-pixel features are correctly indexed to valid-depth pixels."""
        depth = np.zeros((2, 2), dtype=np.float32)
        depth[0, 1] = 1.0
        depth[1, 0] = 2.0
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        feats = np.zeros((2, 2, 4), dtype=np.float32)
        feats[0, 1] = [1, 2, 3, 4]
        feats[1, 0] = [5, 6, 7, 8]
        K = _make_K(100, 100, 0.5, 0.5)

        _, _, out_feats = depth_to_point_cloud(depth, rgb, K, clip_features=feats)

        assert out_feats.shape == (2, 4)
        npt.assert_array_equal(out_feats[0], [1, 2, 3, 4])
        npt.assert_array_equal(out_feats[1], [5, 6, 7, 8])

    def test_roundtrip_project_unproject(self):
        """Project a known 3D point to pixel, unproject → recovers the point."""
        K = _make_K(500, 500, 320, 240)
        pt_cam = np.array([0.5, -0.3, 4.0])

        u, v = _project(pt_cam, K)
        u_int, v_int = int(round(u)), int(round(v))

        H, W = 480, 640
        depth = np.zeros((H, W), dtype=np.float32)
        depth[v_int, u_int] = pt_cam[2]
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        xyz, _, _ = depth_to_point_cloud(depth, rgb, K)

        npt.assert_allclose(xyz[0], pt_cam, atol=0.01)


# ── scale_intrinsics ────────────────────────────────────────────────────────

class TestScaleIntrinsics:
    def test_half_resolution(self):
        K = _make_K(500, 500, 320, 240)
        K_half = scale_intrinsics(K, (240, 320), (480, 640))

        npt.assert_allclose(K_half[0, 0], 250)   # fx halved
        npt.assert_allclose(K_half[1, 1], 250)   # fy halved
        npt.assert_allclose(K_half[0, 2], 160)   # cx halved
        npt.assert_allclose(K_half[1, 2], 120)   # cy halved

    def test_does_not_mutate_input(self):
        K = _make_K(500, 500, 320, 240)
        K_orig = K.copy()
        _ = scale_intrinsics(K, (240, 320), (480, 640))
        npt.assert_array_equal(K, K_orig)

    def test_identity_scaling(self):
        K = _make_K(500, 500, 320, 240)
        K_same = scale_intrinsics(K, (480, 640), (480, 640))
        npt.assert_allclose(K_same, K)

    def test_3x4_matrix(self):
        """scale_intrinsics should work on (3,4) projection matrices too."""
        P = np.zeros((3, 4), dtype=np.float32)
        P[:3, :3] = _make_K(500, 500, 320, 240)
        P_scaled = scale_intrinsics(P, (240, 320), (480, 640))
        assert P_scaled.shape == (3, 4)
        npt.assert_allclose(P_scaled[0, 0], 250)

    def test_scaled_unproject_matches_original(self):
        """Same depth pixel unprojected at original vs scaled resolution
        should produce the same 3D point."""
        K = _make_K(500, 500, 320, 240)
        H, W = 480, 640

        # Place a depth pixel at a known location
        depth = np.zeros((H, W), dtype=np.float32)
        depth[100, 200] = 3.5
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        xyz_full, _, _ = depth_to_point_cloud(depth, rgb, K)

        # Half resolution
        K_half = scale_intrinsics(K, (H // 2, W // 2), (H, W))
        depth_half = cv2.resize(depth, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        rgb_half = np.zeros((H // 2, W // 2, 3), dtype=np.uint8)

        xyz_half, _, _ = depth_to_point_cloud(depth_half, rgb_half, K_half)

        # Should match within discretisation error of one pixel at half-res
        npt.assert_allclose(xyz_half[0], xyz_full[0], atol=0.02)


# ── build_cam_pose ──────────────────────────────────────────────────────────

class TestBuildCamPose:
    def test_identity(self):
        pose = build_cam_pose(np.eye(3), np.zeros(3))
        npt.assert_array_equal(pose, np.eye(4))

    def test_translation_only(self):
        T = np.array([1, 2, 3], dtype=np.float64)
        pose = build_cam_pose(np.eye(3), T)
        npt.assert_array_equal(pose[:3, 3], T)
        npt.assert_array_equal(pose[:3, :3], np.eye(3))

    def test_valid_se3(self):
        """Output should be a valid SE(3) matrix (det(R)==1, last row = [0,0,0,1])."""
        R = _rotation_y(45)
        T = np.array([1, 0, 0.5])
        pose = build_cam_pose(R, T)

        npt.assert_allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-10)
        npt.assert_array_equal(pose[3, :], [0, 0, 0, 1])

    def test_dtype_is_float64(self):
        pose = build_cam_pose(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        assert pose.dtype == np.float64


# ── cam-to-world transform (integration test) ───────────────────────────────

class TestCamToWorld:
    """Test the full chain: depth → camera points → world points."""

    @staticmethod
    def _unproject_and_transform(depth, rgb, K, cam_pose, features=None):
        xyz_cam, colors, feats = depth_to_point_cloud(
            depth, rgb, K, clip_features=features,
        )
        pts_world = (cam_pose[:3, :3] @ xyz_cam.T).T + cam_pose[:3, 3]
        return pts_world, colors, feats

    def test_identity_pose(self):
        """With identity cam-to-world, world == camera points."""
        K = _make_K(100, 100, 1, 1)
        depth = np.array([[0, 0, 0], [0, 5.0, 0], [0, 0, 0]], dtype=np.float32)
        rgb = np.zeros((3, 3, 3), dtype=np.uint8)
        pose = build_cam_pose(np.eye(3), np.zeros(3))

        pts_world, _, _ = self._unproject_and_transform(depth, rgb, K, pose)
        xyz_cam, _, _ = depth_to_point_cloud(depth, rgb, K)

        npt.assert_allclose(pts_world, xyz_cam, atol=1e-10)

    def test_pure_translation(self):
        """Translation shifts all points by T."""
        K = _make_K(100, 100, 1, 1)
        depth = np.array([[0, 0, 0], [0, 2.0, 0], [0, 0, 0]], dtype=np.float32)
        rgb = np.zeros((3, 3, 3), dtype=np.uint8)
        T = np.array([10.0, 20.0, 30.0])
        pose = build_cam_pose(np.eye(3), T)

        pts_world, _, _ = self._unproject_and_transform(depth, rgb, K, pose)
        xyz_cam, _, _ = depth_to_point_cloud(depth, rgb, K)

        npt.assert_allclose(pts_world, xyz_cam + T, atol=1e-10)

    def test_multiview_same_world_point(self):
        """Two cameras observing the same 3D point → same world-space result."""
        pt_world = np.array([1.0, 2.0, 5.0])
        K = _make_K(500, 500, 320, 240)
        H, W = 480, 640

        pose1 = build_cam_pose(np.eye(3), np.zeros(3))
        pose2 = build_cam_pose(_rotation_y(15), np.array([0.3, 0.0, 0.2]))

        recovered = []
        for pose in [pose1, pose2]:
            R = pose[:3, :3]
            T = pose[:3, 3]
            # cam-to-world: pt_world = R @ pt_cam + T
            # world-to-cam: pt_cam = R^T @ (pt_world - T)
            pt_cam = R.T @ (pt_world - T)
            if pt_cam[2] <= 0:
                continue

            u, v = _project(pt_cam.astype(np.float32), K)
            u_int, v_int = int(round(u)), int(round(v))
            if not (0 <= u_int < W and 0 <= v_int < H):
                continue

            depth = np.zeros((H, W), dtype=np.float32)
            depth[v_int, u_int] = float(pt_cam[2])
            rgb = np.zeros((H, W, 3), dtype=np.uint8)

            xyz_cam, _, _ = depth_to_point_cloud(depth, rgb, K)
            pt_rec = (R @ xyz_cam.T).T + T
            recovered.append(pt_rec[0])

        assert len(recovered) == 2, "Both cameras should see the point"
        npt.assert_allclose(recovered[0], pt_world, atol=0.02)
        npt.assert_allclose(recovered[1], pt_world, atol=0.02)


# ── SparseVoxelGrid ─────────────────────────────────────────────────────────

class TestSparseVoxelGrid:
    def test_single_point(self):
        grid = SparseVoxelGrid(voxel_size=1.0, origin=np.zeros(3))
        grid.integrate(
            np.array([[0.5, 0.5, 0.5]]),
            np.array([[1.0, 2.0]]),
        )
        assert len(grid) == 1
        npt.assert_allclose(grid.get_centroids(), [[0.5, 0.5, 0.5]])
        npt.assert_allclose(grid.get_features(), [[1.0, 2.0]])
        npt.assert_array_equal(grid.get_counts(), [1])

    def test_two_points_same_voxel(self):
        grid = SparseVoxelGrid(voxel_size=1.0, origin=np.zeros(3))
        pts = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])
        feats = np.array([[1.0, 0.0], [0.0, 1.0]])
        grid.integrate(pts, feats)

        assert len(grid) == 1
        npt.assert_allclose(grid.get_centroids()[0], [0.5, 0.5, 0.5])
        npt.assert_allclose(grid.get_features()[0], [0.5, 0.5])
        npt.assert_array_equal(grid.get_counts(), [2])

    def test_two_points_different_voxels(self):
        grid = SparseVoxelGrid(voxel_size=1.0, origin=np.zeros(3))
        pts = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
        feats = np.array([[1.0], [2.0]])
        grid.integrate(pts, feats)

        assert len(grid) == 2

    def test_multi_batch_running_average(self):
        """Features and centroids averaged correctly across two integrate calls."""
        grid = SparseVoxelGrid(voxel_size=1.0, origin=np.zeros(3))

        # Batch 1: two points
        grid.integrate(
            np.array([[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        )
        # Batch 2: one more point in the same voxel
        grid.integrate(
            np.array([[0.5, 0.5, 0.5]]),
            np.array([[0.5, 0.5]]),
        )

        assert len(grid) == 1
        npt.assert_array_equal(grid.get_counts(), [3])
        npt.assert_allclose(grid.get_centroids()[0], [0.5, 0.5, 0.5])
        npt.assert_allclose(grid.get_features()[0], [0.5, 0.5])

    def test_empty_input(self):
        grid = SparseVoxelGrid(voxel_size=0.5)
        grid.integrate(np.empty((0, 3)), np.empty((0, 4)))
        assert len(grid) == 0

    def test_lazy_origin(self):
        """Origin is set from first batch when not provided."""
        grid = SparseVoxelGrid(voxel_size=1.0)
        pts = np.array([[10.0, 20.0, 30.0]])
        feats = np.array([[1.0]])
        grid.integrate(pts, feats)

        assert grid.origin is not None
        assert len(grid) == 1
        npt.assert_allclose(grid.get_centroids()[0], [10.0, 20.0, 30.0])
