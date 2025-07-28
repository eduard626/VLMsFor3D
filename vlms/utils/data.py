import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional


def load_tum_rgb_data(
    base_path: str,
    camera_calibration_file: Optional[str],
    stamp_threshold: float = 0.02,
) -> dict:
    # RGB image
    rgb_data_path = os.path.join(base_path, "rgb")
    if not os.path.exists(rgb_data_path):
        raise FileNotFoundError(
            f"Data path {rgb_data_path} does not exist. Please check the path to your dataset."
        )
    image_files = sorted(
        os.listdir(rgb_data_path), key=lambda x: float(x.split(".png")[0])
    )
    image_files = [f for f in image_files if f.endswith(".png") or f.endswith(".jpg")]

    depth_data_path = os.path.join(base_path, "depth")
    if not os.path.exists(depth_data_path):
        raise FileNotFoundError(
            f"Depth path {depth_data_path} does not exist. Please check the path to your dataset."
        )
    depth_files = sorted(
        os.listdir(depth_data_path), key=lambda x: float(x.split(".png")[0])
    )
    depth_files = [f for f in depth_files if f.endswith(".png")]

    # we need to match the images with the depth files
    # as they don't have the same timestamps, see https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools
    # the idea is to find matching stamps under some threshold time difference

    stamp_matches = {}
    rgb_keys = set([f.split(".png")[0] for f in image_files])
    depth_keys = set([f.split(".png")[0] for f in depth_files])
    ref_keys = rgb_keys if len(rgb_keys) < len(depth_keys) else depth_keys
    target_keys = depth_keys if len(rgb_keys) < len(depth_keys) else rgb_keys
    for key in ref_keys:
        candidates = [
            t for t in target_keys if abs(float(t) - float(key)) < stamp_threshold
        ]
        if len(candidates) > 0:
            candidates = sorted(candidates, key=lambda x: abs(float(x) - float(key)))
            # take the closest one as long as is not already matched
            for candidate in candidates:
                if candidate not in stamp_matches:
                    stamp_matches[key] = candidate
                    break
    if len(stamp_matches) == 0:
        raise ValueError(
            "No matching stamps found. Please check the stamp threshold or the dataset."
        )

    # we don't know which is RGB and which is depth
    if len(rgb_keys) < len(depth_keys):
        # rgb is the reference
        image_files = [f"{key}.png" for key in stamp_matches.keys()]
        depth_files = [f"{stamp_matches[key]}.png" for key in stamp_matches.keys()]
    else:
        # depth is the reference
        depth_files = [f"{key}.png" for key in stamp_matches.keys()]
        image_files = [f"{stamp_matches[key]}.png" for key in stamp_matches.keys()]

    if len(image_files) != len(depth_files):
        raise ValueError(
            "The number of images and depth files do not match. Please check the dataset."
        )

    # the GT poses
    gt_poses_path = os.path.join(base_path, "groundtruth.txt")
    gt_T = []
    gt_R = []
    stamps = []
    margin = 0.02  # margin to avoid floating point precision issues
    with open(gt_poses_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            stamps.append(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            rot_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
            gt_T.append([tx, ty, tz])
            gt_R.append(rot_mat)

    # filter poses by target timestamps
    if len(rgb_keys) < len(depth_keys):
        ref_stamps = [f.split(".png")[0] for f in image_files]
    else:
        ref_stamps = [f.split(".png")[0] for f in depth_files]
    indices = []
    stamps = np.array(stamps, dtype=np.float64)
    for i in range(len(ref_stamps)):
        stamp = float(ref_stamps[i])
        # find the closest timestamp in the ground truth poses
        closest_index = np.argmin(np.abs(stamps - stamp))
        closest_stamp = stamps[closest_index]
        if abs(closest_stamp - stamp) < margin:
            indices.append(closest_index)
    # sanity check: indices are not empty and are not repeated?
    if len(indices) == 0:
        raise ValueError(
            "No matching ground truth poses found. Please check the dataset."
        )

    assert (
        indices[0] != indices[len(indices) // 2] and indices[0] != indices[-1]
    ), "Indices should not be repeated."

    gt_R = [gt_R[i] for i in indices]
    gt_T = [gt_T[i] for i in indices]

    gt_T = np.array(gt_T, dtype=np.float32)
    gt_R = np.array(gt_R, dtype=np.float32)

    # intrinsic matrix for the camera
    # according to the dataset docs, depth images are already registered to RGB images
    # https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
    if camera_calibration_file is not None and os.path.exists(camera_calibration_file):
        raise NotImplementedError(
            "Camera calibration file loading is not implemented yet."
        )
    else:
        # HARDCODED INTRINSICS for sequence desk2
        # @TODO: implement loading from file
        K = [
            517.306408,
            0.000000,
            318.643040,
            0.000000,
            516.469215,
            255.313989,
            0.000000,
            0.000000,
            1.000000,
        ]
        K = np.array(K, dtype=np.float32).reshape(3, 3)
        distortion = [0.262383, -0.953104, -0.005358, 0.002628, 1.163314]
        distortion = np.array(distortion, dtype=np.float32).reshape(1, 5)
        P = [
            546.024414,
            0.000000,
            319.711258,
            0.000000,
            0.000000,
            542.211182,
            251.374926,
            0.0000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]
        P = np.array(P, dtype=np.float32).reshape(
            3, 4
        )  # includes K for rectified images

    data_out = {
        "rgb": image_files,
        "depth": depth_files,
        "stamp_matches": stamp_matches,
        "poses": {
            "R": gt_R,
            "T": gt_T,
        },
        "camera_intrinsics": {
            "K": K,
            "distortion": distortion,
            "P": P,
        },
    }
    return data_out
