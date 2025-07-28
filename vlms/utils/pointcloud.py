import numpy as np
import open3d as o3d


def depth_to_point_cloud(
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    intrinsic_matrix: np.ndarray,
    clip_features: np.ndarray = None,
) -> np.ndarray:
    """
    Convert a depth image to a point cloud.

    Args:
        depth_image (np.ndarray): Depth image of shape (H, W).
        rgb_image (np.ndarray): RGB image of shape (H, W, 3).
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        clip_features (np.ndarray, optional): CLIP features of shape (H, W, C) to be included in the point cloud.

    Returns:
        np.ndarray: Point cloud of shape (N, 3), where N is the number of valid points.
    """
    # assume depth image has been converted to meters
    assert depth_image.dtype == np.float32, "Depth image should be of type float32"
    assert (
        rgb_image.dtype == np.uint8 and rgb_image.shape[:2] == depth_image.shape
    ), "RGB image should be of type uint8"
    assert intrinsic_matrix.shape == (
        3,
        3,
    ), "Intrinsic matrix should be of shape (3, 3)"

    h, w = depth_image.shape
    y_indices, x_indices = np.indices((h, w))

    # Flatten the indices and depth image
    x_flat = x_indices.flatten()
    y_flat = y_indices.flatten()
    z_flat = depth_image.flatten()
    r = rgb_image[:, :, 0].flatten()  # Red channel
    g = rgb_image[:, :, 1].flatten()  # Green channel
    b = rgb_image[:, :, 2].flatten()  # Blue channel

    # Filter out invalid points (depth == 0)
    valid_mask = z_flat > 0
    x_flat = x_flat[valid_mask]
    y_flat = y_flat[valid_mask]
    z_flat = z_flat[valid_mask]
    r = r[valid_mask]
    g = g[valid_mask]
    b = b[valid_mask]
    if clip_features is not None:
        # HARDCODED for CLIP model with 768 features per pixel
        assert clip_features.shape == (
            h,
            w,
            768,
        ), f"CLIP features should be of shape (H, W, 3) but got {clip_features.shape}"
        clip_features_flat = clip_features.reshape(-1, clip_features.shape[-1])[
            valid_mask
        ]

    # Convert to camera coordinates
    x_camera = (x_flat - intrinsic_matrix[0, 2]) * z_flat / intrinsic_matrix[0, 0]
    y_camera = (y_flat - intrinsic_matrix[1, 2]) * z_flat / intrinsic_matrix[1, 1]

    # Stack to create point cloud
    if clip_features is not None:
        point_cloud = np.stack((x_camera, y_camera, z_flat, r, g, b), axis=-1)
        point_cloud = np.concatenate((point_cloud, clip_features_flat), axis=-1)
    else:
        point_cloud = np.stack((x_camera, y_camera, z_flat, r, g, b), axis=-1)

    return point_cloud


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
    point_cloud = depth_to_point_cloud(
        depth_image, color_image, camera_intrinsics, clip_features
    )

    # Convert point cloud to homogeneous coordinates
    ones = np.ones((point_cloud.shape[0], 1))
    point_cloud_homogeneous = np.hstack((point_cloud[:, :3], ones))  # Keep only XYZ

    # Apply camera pose transformation
    posed_point_cloud = point_cloud_homogeneous @ camera_pose.T

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(posed_point_cloud[:, :3])  # Keep only XYZ
    pc.colors = o3d.utility.Vector3dVector(
        point_cloud[:, 3:6] / 255.0
    )  # Normalize RGB values to [0, 1]

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
