#!/usr/bin/env python3
"""FindAnything-style mapping pipeline: eSAM segmentation + CLIP dense features + 3D voxel fusion.

Processes TUM RGB-D sequences frame-by-frame, skipping blurry frames via a
time-window policy, and fuses per-pixel CLIP features into a global voxel map.
"""

import argparse
import os
import resource
import sys
import time
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

import cv2
import clip
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Resolve project root so we can find eTAM and vlms regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ETAM_DIR = PROJECT_ROOT / "eTAM"
if str(ETAM_DIR) not in sys.path:
    sys.path.insert(0, str(ETAM_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from efficient_track_anything.build_efficienttam import build_efficienttam
from efficient_track_anything.automatic_mask_generator import (
    EfficientTAMAutomaticMaskGenerator,
)

from vlms.utils.data import compute_blur_score, load_frame, load_tum_rgb_data
from vlms.utils.features import extract_clip_image_features
from vlms.utils.pointcloud import (
    SparseVoxelGrid,
    build_cam_pose,
    create_posed_cloud,
    depth_to_point_cloud,
    scale_intrinsics,
)
from vlms.utils.visualisation import draw_anns


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FindAnything mapping pipeline — eSAM + CLIP + voxel fusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--data-path", type=str, required=True,
        help="Path to a TUM RGB-D dataset directory",
    )
    parser.add_argument(
        "-b", "--blur-threshold", type=float, default=100.0,
        help="Minimum Laplacian-variance score to accept a frame as sharp",
    )
    parser.add_argument(
        "-t", "--time-window", type=float, default=2.0,
        help="Time window (seconds). If no sharp frame is found within the "
             "window the best-scored frame is used instead",
    )
    parser.add_argument(
        "-v", "--voxel-resolution", type=float, default=0.03,
        help="Voxel edge length in metres for feature fusion",
    )
    parser.add_argument(
        "-m", "--max-views", type=int, default=0,
        help="Maximum number of frames to process (0 = all)",
    )
    parser.add_argument(
        "-q", "--query", type=str, default=None,
        help="Optional text query for similarity visualisation",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="./output",
        help="Directory for saving results (point clouds, screenshots)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug outputs: blur scores, mask images, timing stats",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Open an interactive Open3D viewer (requires a display)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(42)
    print(f"Using device: {device}")
    return device


def setup_models(device: torch.device):
    """Load eSAM and CLIP models, build the mask generator."""
    checkpoint = str(ETAM_DIR / "checkpoints" / "efficienttam_ti_512x512.pt")
    model_cfg = "configs/efficienttam/efficienttam_ti_512x512.yaml"

    print("Loading eTAM model …")
    efficienttam = build_efficienttam(model_cfg, checkpoint, device=device)

    print("Loading CLIP ViT-L/14@336px …")
    clip_model, clip_transforms = clip.load("ViT-L/14@336px", device=device, jit=False)

    # Prompt-point grid (8×8, normalised to [0, 1])
    grid_side = 8
    # Use a dummy size for normalisation — the grid is resolution-independent
    x = [(i + 0.5) / grid_side for i in range(grid_side)]
    y = [(i + 0.5) / grid_side for i in range(grid_side)]
    input_points = np.array(np.meshgrid(x, y)).reshape(2, -1).T.astype(np.float32)

    mask_generator = EfficientTAMAutomaticMaskGenerator(
        model=efficienttam,
        points_per_batch=64,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.9,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        point_grids=[input_points, input_points],
        points_per_side=None,
        crop_overlap_ratio=0.25,
    )

    # Warmup with a small dummy image
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    mask_generator.generate(dummy)

    return {
        "efficienttam": efficienttam,
        "clip_model": clip_model,
        "clip_transforms": clip_transforms,
        "mask_generator": mask_generator,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(args: argparse.Namespace) -> dict:
    print(f"Loading dataset from {args.data_path} …")
    return load_tum_rgb_data(
        base_path=args.data_path,
        stamp_threshold=0.02,
        camera_calibration_file=None,
    )


# ---------------------------------------------------------------------------
# Per-frame CLIP feature extraction (shared helper)
# ---------------------------------------------------------------------------

def extract_dense_clip_features(
    rgb_image: np.ndarray,
    clip_model,
    clip_transforms,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    """Return dense CLIP features at rgb_image resolution.

    Returns:
        features_np: (H, W, 768) numpy array, min-max scaled to [0, 1] for voxel fusion.
        dense_norm:  (1, C, H, W) torch tensor, L2-normalised along C — use for similarity maps.
    """
    h_orig, w_orig = rgb_image.shape[:2]

    # CLIP expects square input
    target_size = min(h_orig, w_orig)
    rgb_square = cv2.resize(rgb_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    input_tensor = clip_transforms(Image.fromarray(rgb_square)).unsqueeze(0).to(device)

    with torch.no_grad():
        clip_out = clip_model.encode_image(input_tensor, dense_features=True)

    dense = extract_clip_image_features(clip_out, input_tensor.shape[-2:])  # (1, C, H, W)

    # Restore original aspect ratio (image was squished to square, expand width back)
    original_ratio = w_orig / h_orig
    if abs(original_ratio - 1.0) > 1e-3:
        feat_h, feat_w = dense.shape[2:4]
        new_w = int(feat_w * original_ratio)
        dense = F.interpolate(dense, size=(feat_h, new_w), mode="bilinear", align_corners=False)

    # L2-normalised tensor for similarity computation
    dense_norm = F.normalize(dense, dim=1)

    features_np = dense.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    features_np = (features_np - features_np.min()) / (features_np.max() - features_np.min() + 1e-8)
    return features_np, dense_norm


# ---------------------------------------------------------------------------
# Time-window blur gating
# ---------------------------------------------------------------------------

def _timestamp_from_filename(filename: str) -> float:
    """Extract the floating-point timestamp from a TUM-style filename like '1305031910.765238.png'."""
    return float(filename.split(".png")[0])


def _compute_blur_scores_parallel(
    image_files: list[str], data_path: str, max_workers: int = 8,
) -> list[float]:
    """Compute blur scores for all images using threaded I/O."""

    def _score_one(img_file: str) -> float:
        img_path = os.path.join(data_path, "rgb", img_file)
        return compute_blur_score(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_score_one, image_files))


def select_frames_to_process(
    image_files: list[str],
    data_path: str,
    blur_threshold: float,
    time_window: float,
    debug: bool,
) -> list[int]:
    """Return the indices of frames that should be processed.

    Policy — within each time window:
      * Accept the first frame whose blur score >= blur_threshold.
      * If the window expires without an accepted frame, pick the sharpest
        frame seen so far in that window.
    """
    # Phase 1: parallel blur scoring (I/O-bound — threads overlap disk reads)
    scores = _compute_blur_scores_parallel(image_files, data_path)

    # Phase 2: sequential window policy over pre-computed scores
    selected: list[int] = []

    window_start_time: float | None = None
    best_in_window_idx: int | None = None
    best_in_window_score: float = -1.0

    for i, img_file in enumerate(image_files):
        ts = _timestamp_from_filename(img_file)
        score = scores[i]

        # Start a new window on the very first frame
        if window_start_time is None:
            window_start_time = ts
            best_in_window_idx = None
            best_in_window_score = -1.0

        # Track the best frame in the current window
        if best_in_window_idx is None or score > best_in_window_score:
            best_in_window_idx = i
            best_in_window_score = score

        elapsed = ts - window_start_time

        if score >= blur_threshold:
            # Sharp enough — accept and reset the window
            selected.append(i)
            if debug:
                print(f"  [ACCEPT] frame {i} ({img_file}) — score {score:.1f} >= {blur_threshold}")
            window_start_time = None
        elif elapsed >= time_window:
            # Window expired — fall back to the best frame we saw
            selected.append(best_in_window_idx)
            if debug:
                print(
                    f"  [FALLBACK] frame {best_in_window_idx} ({image_files[best_in_window_idx]}) "
                    f"— best score {best_in_window_score:.1f} (window expired after {elapsed:.2f}s)"
                )
            window_start_time = None
        else:
            if debug:
                print(f"  [SKIP] frame {i} ({img_file}) — score {score:.1f}, window {elapsed:.2f}s")

    # Handle any remaining window that never closed
    if window_start_time is not None and best_in_window_idx is not None:
        selected.append(best_in_window_idx)
        if debug:
            print(
                f"  [FALLBACK-END] frame {best_in_window_idx} "
                f"({image_files[best_in_window_idx]}) — best score {best_in_window_score:.1f}"
            )

    return selected


# ---------------------------------------------------------------------------
# Core processing loop
# ---------------------------------------------------------------------------

def process_frames(args, models, data, device):
    """Run the full per-frame pipeline: segment → CLIP features → 3D voxel fusion."""
    image_files = data["rgb"]
    depth_files = data["depth"]
    gt_poses = data["poses"]
    camera_P = data["camera_intrinsics"]["P"]

    clip_model = models["clip_model"]
    clip_transforms = models["clip_transforms"]
    mask_generator = models["mask_generator"]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.debug:
        os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "debug"), exist_ok=True)
        if args.query:
            os.makedirs(os.path.join(args.output_dir, "similarity"), exist_ok=True)

    # Pre-encode text query once (used for per-frame similarity maps in debug mode)
    text_features = None
    if args.debug and args.query:
        tokens = clip.tokenize([args.query]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    # ---- blur gating ---------------------------------------------------------
    # Not every frame has a matching GT pose (timestamp proximity filtering in
    # load_tum_rgb_data), so cap to the number of available poses.
    n_with_poses = len(gt_poses["R"])
    print("Scoring frames for blur …")
    total_frames = min(len(image_files), n_with_poses)
    if args.max_views > 0:
        total_frames = min(total_frames, args.max_views)
    candidate_files = image_files[:total_frames]

    selected_indices = select_frames_to_process(
        candidate_files,
        args.data_path,
        blur_threshold=args.blur_threshold,
        time_window=args.time_window,
        debug=args.debug,
    )
    print(f"Selected {len(selected_indices)}/{total_frames} frames to process")

    # ---- sparse voxel grid (surface-conforming) ------------------------------
    voxel_grid = SparseVoxelGrid(voxel_size=args.voxel_resolution)
    print(f"Sparse voxel grid at {args.voxel_resolution}m resolution")

    # ---- per-frame feature fusion -------------------------------------------
    all_pts_world: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    P_scaled = None  # lazily computed on first frame

    for frame_idx in tqdm(selected_indices, desc="Processing frames"):
        t0 = time.time()

        rgb_img, depth_img = load_frame(
            args.data_path, image_files[frame_idx], depth_files[frame_idx],
        )

        # -- segmentation masks ------------------------------------------------
        masks = mask_generator.generate(rgb_img)

        if args.debug:
            mask_rgba = draw_anns(masks, borders=True)  # (H, W, 4) float RGBA
            rgb_float = rgb_img.astype(np.float32) / 255.0
            alpha = mask_rgba[:, :, 3:4]
            overlay = rgb_float * (1 - alpha) + mask_rgba[:, :, :3] * alpha
            overlay = np.clip(overlay, 0, 1)
            plt.imsave(
                os.path.join(args.output_dir, "masks", f"frame_{frame_idx:05d}.png"),
                overlay,
            )

        # -- CLIP dense features -----------------------------------------------
        clip_features_np, dense_tensor = extract_dense_clip_features(
            rgb_img, clip_model, clip_transforms, device,
        )

        # -- per-frame similarity map (debug + query) --------------------------
        if args.debug and text_features is not None:
            # Negated cosine similarity (CLS-token leakage — see CLIP Surgery)
            sim_map = -torch.einsum("bchw,bc->bhw", dense_tensor, text_features)
            sim = sim_map.squeeze(0).cpu().numpy()
            sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)

            # Resize sim map to match RGB so both panels are identical in size
            sim_display = cv2.resize(sim.astype(np.float32),
                                     (rgb_img.shape[1], rgb_img.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(rgb_img)
            axes[0].set_title("RGB")
            axes[0].axis("off")
            bar = axes[1].imshow(sim_display, cmap="turbo", vmin=0, vmax=1)
            axes[1].set_title(f"Similarity: '{args.query}'")
            axes[1].axis("off")
            # Layout first, then place colorbar in its own axis using
            # the computed position of the similarity panel.
            fig.tight_layout(rect=[0, 0.08, 1, 1])  # leave 8% bottom margin
            pos = axes[1].get_position()
            cbar_ax = fig.add_axes([pos.x0, 0.03, pos.width, 0.03])
            fig.colorbar(bar, cax=cbar_ax, orientation="horizontal")
            fig.savefig(
                os.path.join(args.output_dir, "similarity", f"frame_{frame_idx:05d}.png"),
                dpi=120, bbox_inches="tight",
            )
            plt.close(fig)

        # Build segment mask (union of all eSAM masks)
        seg_mask = np.zeros(rgb_img.shape[:2], dtype=bool)
        for m in masks:
            seg_mask |= m["segmentation"]

        # Down-sample RGB/depth/mask to match CLIP feature resolution
        feat_h, feat_w = clip_features_np.shape[:2]
        rgb_ds = cv2.resize(rgb_img, (feat_w, feat_h), interpolation=cv2.INTER_LINEAR)
        depth_ds = cv2.resize(depth_img, (feat_w, feat_h), interpolation=cv2.INTER_NEAREST)
        seg_mask_ds = cv2.resize(
            seg_mask.astype(np.uint8), (feat_w, feat_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        if P_scaled is None:
            P_scaled = scale_intrinsics(
                camera_P, (feat_h, feat_w), rgb_img.shape[:2],
            )

        cam_pose = build_cam_pose(gt_poses["R"][frame_idx], gt_poses["T"][frame_idx])

        # Project ALL valid depth → complete geometry
        xyz_cam, rgb_vals, feats = depth_to_point_cloud(
            depth_ds.astype(np.float32), rgb_ds,
            P_scaled[:3, :3], clip_features=clip_features_np,
        )

        pts_world = (cam_pose[:3, :3] @ xyz_cam.T).T + cam_pose[:3, 3]

        # Collect all points for RGB cloud (complete surface)
        all_pts_world.append(pts_world)
        all_colors.append(rgb_vals / 255.0)

        # Fuse features only for pixels within a segment — unsegmented
        # pixels don't belong to a semantic unit and would dilute the
        # voxel features with noise.
        seg_flat = seg_mask_ds.ravel()[depth_ds.ravel() > 0]
        if seg_flat.any():
            voxel_grid.integrate(pts_world[seg_flat], feats[seg_flat])

        if args.debug:
            dt = time.time() - t0
            print(f"  Frame {frame_idx}: {len(masks)} masks, "
                  f"{len(xyz_cam)} points, {dt:.2f}s")

    # Build final RGB point cloud (single Open3D construction + downsample)
    big_cloud = o3d.geometry.PointCloud()
    if all_pts_world:
        big_cloud.points = o3d.utility.Vector3dVector(np.concatenate(all_pts_world))
        big_cloud.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors))
        big_cloud = big_cloud.voxel_down_sample(0.01)

    print(f"Sparse voxel grid: {len(voxel_grid)} occupied voxels")
    n_frames_processed = len(selected_indices)
    return big_cloud, voxel_grid, n_frames_processed


# ---------------------------------------------------------------------------
# Query & visualisation
# ---------------------------------------------------------------------------

def query_features(
    query_text: str,
    clip_model,
    voxel_grid: SparseVoxelGrid,
    device: torch.device,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Compute per-voxel similarity to a text query. Returns coloured cloud + scores."""
    voxel_feats = voxel_grid.get_features()
    centroids = voxel_grid.get_centroids()

    tokens = clip.tokenize([query_text]).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
    text_feats = F.normalize(text_feats, dim=-1).cpu().numpy()

    # Negated cosine similarity (CLS-token leakage — see CLIP Surgery)
    scores = -cosine_similarity(voxel_feats, text_feats).flatten()
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    colors = cm.viridis(scores)[:, :3]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(centroids)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud, scores


def save_results(
    args: argparse.Namespace,
    big_cloud: o3d.geometry.PointCloud,
    voxel_grid: SparseVoxelGrid,
    clip_model,
    device: torch.device,
):
    os.makedirs(args.output_dir, exist_ok=True)

    # Save RGB point cloud
    ply_path = os.path.join(args.output_dir, "cloud.ply")
    o3d.io.write_point_cloud(ply_path, big_cloud)
    print(f"Saved point cloud to {ply_path}")

    # Build activated-voxels cloud (coloured by observation count)
    if len(voxel_grid) == 0:
        print("Warning: no voxels were activated — nothing to save beyond the raw cloud.")
        return

    centroids = voxel_grid.get_centroids()
    counts = voxel_grid.get_counts()
    counts_norm = (counts - counts.min()) / (counts.max() - counts.min() + 1e-8)
    colors = cm.viridis(counts_norm)[:, :3]

    activated_cloud = o3d.geometry.PointCloud()
    activated_cloud.points = o3d.utility.Vector3dVector(centroids)
    activated_cloud.colors = o3d.utility.Vector3dVector(colors)

    voxel_ply = os.path.join(args.output_dir, "voxels.ply")
    o3d.io.write_point_cloud(voxel_ply, activated_cloud)
    print(f"Saved voxel cloud ({len(voxel_grid)} voxels) to {voxel_ply}")

    # Optional query visualisation
    if args.query:
        print(f"Running query: '{args.query}' …")
        query_cloud, scores = query_features(
            args.query, clip_model, voxel_grid, device,
        )
        query_ply = os.path.join(args.output_dir, "query.ply")
        o3d.io.write_point_cloud(query_ply, query_cloud)
        print(f"Saved query cloud to {query_ply}")

        # Offscreen screenshot
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(width=640, height=480, visible=False)
        viewer.add_geometry(query_cloud)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        viewer.add_geometry(origin)
        viewer.poll_events()
        viewer.update_renderer()
        viewer.get_view_control().set_zoom(0.3)
        screenshot = np.asarray(viewer.capture_screen_float_buffer(do_render=True))

        screenshot_path = os.path.join(args.output_dir, "query_screenshot.png")
        plt.imsave(screenshot_path, screenshot)
        print(f"Saved query screenshot to {screenshot_path}")
        viewer.destroy_window()

        if args.interactive:
            o3d.visualization.draw_geometries(
                [query_cloud, origin],
                window_name=f"Query: {args.query}",
                width=800, height=600,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _format_bytes(n_bytes: int) -> str:
    """Format byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _log_memory_summary(t_start: float, n_frames: int, n_voxels: int):
    """Print a memory and timing summary to the console."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = rusage.ru_maxrss * 1024  # ru_maxrss is in KB on Linux
    elapsed = time.time() - t_start

    print("\n--- Memory & Timing Summary ---")
    print(f"  Peak RSS:            {_format_bytes(peak_rss)}")
    print(f"  Frames processed:    {n_frames}")
    print(f"  Activated voxels:    {n_voxels}")
    sparse_mem = n_voxels * 768 * 4  # float32 features per voxel
    print(f"  Voxel features mem:  {_format_bytes(sparse_mem)} (estimated, float32)")
    print(f"  Total wall time:     {elapsed:.1f}s")
    if n_frames > 0:
        print(f"  Per-frame average:   {elapsed / n_frames:.2f}s")
    if torch.cuda.is_available():
        gpu_peak = torch.cuda.max_memory_allocated()
        gpu_reserved = torch.cuda.max_memory_reserved()
        print(f"  GPU peak allocated:  {_format_bytes(gpu_peak)}")
        print(f"  GPU peak reserved:   {_format_bytes(gpu_reserved)}")
    print("-------------------------------")


def main():
    t_start = time.time()
    args = parse_args()
    device = setup_device()

    models = setup_models(device)
    data = load_dataset(args)

    big_cloud, voxel_grid, n_frames = process_frames(
        args, models, data, device,
    )

    save_results(
        args, big_cloud, voxel_grid,
        models["clip_model"], device,
    )

    _log_memory_summary(t_start, n_frames, len(voxel_grid))
    print("Done.")


if __name__ == "__main__":
    main()
