import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Poor man's implementation of FindAnything
    ## Mapping - part 1
    I'm experimenting with [Open-Vocabulary and Object-Centric Mapping for Robot Exploration in Any Environment](https://arxiv.org/pdf/2504.08603). At least until they release the code, I will try to implement some bits.

    In this notebook, I explore the "mapping" portion as shown in the diagram below. Particularly:
    1. Over-segmentation of an RGB image via eSAM, and
    2. Computing SigLIP2 (dense) features for the segments, and
    3. Projecting/fusing into a 3D representation*

    *For now, I will use GT pose info for projection into 3D, FindAnything uses VI SLAM with [supereight2](https://github.com/ethz-mrl/supereight2)

    ![](../images/findanything.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First thing is setting up eSAM and SigLIP2 models.
    * For eSAM we use the [Efficient Track Anything](https://github.com/yformer/EfficientTAM) version, as I couldn't make the [original eSAM](https://github.com/yformer/EfficientSAM) work.
    * For the vision-language model, we use Google's [SigLIP2](https://arxiv.org/abs/2502.14786) (so400m-patch14-384) via HuggingFace transformers. SigLIP2 uses sigmoid contrastive loss and is trained with self-distillation for better dense/local features.

    eTAM should be a submodule in this repo (see README.md)
    """)
    return


@app.cell
def _():
    import torch
    import sys
    import os
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import torch.nn.functional as F

    # Resolve project root so imports work regardless of cwd
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    # SigLIP2 via HuggingFace transformers
    from transformers import AutoModel, AutoProcessor

    # add the submodule path to sys.path
    _etam_path = os.path.join(_project_root, "eTAM")
    if _etam_path not in sys.path:
        sys.path.append(_etam_path)

    try:
        from efficient_track_anything.build_efficienttam import build_efficienttam
    except ImportError:
        raise ImportError("Efficient Track Anything could not be installed, please check the installation instructions in the README.md file")

    from efficient_track_anything.automatic_mask_generator import EfficientTAMAutomaticMaskGenerator
    from vlms.utils.data import load_tum_rgb_data
    from vlms.utils.visualisation import draw_anns
    from vlms.utils.features import extract_dense_visual_features, pca_on_image_features
    from vlms.utils.pointcloud import create_posed_cloud, depth_to_point_cloud

    import open3d as o3d
    from tqdm import tqdm
    import cv2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

    project_root = _project_root
    return (
        AutoModel,
        AutoProcessor,
        EfficientTAMAutomaticMaskGenerator,
        F,
        Image,
        build_efficienttam,
        cm,
        create_posed_cloud,
        cv2,
        depth_to_point_cloud,
        device,
        draw_anns,
        extract_dense_visual_features,
        load_tum_rgb_data,
        np,
        o3d,
        os,
        pca_on_image_features,
        plt,
        project_root,
        torch,
        tqdm,
    )


@app.cell
def _(build_efficienttam, device, os, project_root):
    checkpoint = os.path.join(project_root, "eTAM/checkpoints/efficienttam_ti_512x512.pt")
    model_cfg = "configs/efficienttam/efficienttam_ti_512x512.yaml"
    efficienttam = build_efficienttam(model_cfg, checkpoint, device=device)
    return (efficienttam,)


@app.cell
def _(AutoModel, AutoProcessor, device):
    # SigLIP2 so400m-patch14-384: sigmoid contrastive loss, trained for dense features
    # ~1.6GB download on first run, cached in ~/.cache/huggingface
    siglip2_model_id = "google/siglip2-so400m-patch14-384"
    siglip2_model = AutoModel.from_pretrained(siglip2_model_id).to(device).eval()
    siglip2_processor = AutoProcessor.from_pretrained(siglip2_model_id)
    siglip2_patch_size = siglip2_model.vision_model.config.patch_size  # 14
    return (siglip2_model, siglip2_processor, siglip2_patch_size)


@app.cell
def _(load_tum_rgb_data):
    data_path = "/home/eduardo/Downloads/rgbd_dataset_freiburg1_desk2"
    data = load_tum_rgb_data(base_path=data_path, stamp_threshold=0.02, camera_calibration_file=None)
    image_files = data["rgb"]
    depth_files = data["depth"]
    gt_poses = data["poses"]
    camera_K = data["camera_intrinsics"]["K"]
    camera_P = data["camera_intrinsics"]["P"]
    return (camera_P, data_path, depth_files, gt_poses, image_files)


@app.cell
def _(Image, data_path, image_files, np, os, plt):
    # Visualize sample images from the dataset
    print("Final image files:", len(image_files), "first 5 files:", image_files[:5])

    _, _ax = plt.subplots(1, 5, figsize=(15, 5))
    _sample_ids = np.linspace(0, len(image_files) - 1, 5).astype(int)
    for _i, _image_file_id in enumerate(_sample_ids):
        _image_file = image_files[_image_file_id]
        _image_path = os.path.join(data_path, "rgb", _image_file)
        assert os.path.exists(_image_path), f"Image file {_image_path} does not exist"
        _image = Image.open(_image_path).convert("RGB")
        _w, _h = _image.size
        _target_h = 512
        _scale = _target_h / _h
        _w = int(_w * _scale)
        _h = int(_h * _scale)
        _image = _image.resize((_w, _h), Image.Resampling.LANCZOS)
        _ax[_i].imshow(_image)
        _ax[_i].set_title(f"{_image_file}")
        _ax[_i].axis("off")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(
    EfficientTAMAutomaticMaskGenerator,
    Image,
    data_path,
    draw_anns,
    efficienttam,
    image_files,
    np,
    os,
    plt,
):
    # grid of prompt points: FindAnything uses 8x8 for simulation and 5x5 for real images
    # Compute grid on a reference image to get w, h
    _ref_image = Image.open(os.path.join(data_path, "rgb", image_files[0])).convert("RGB")
    _w, _h = _ref_image.size
    _target_h = 512
    _scale = _target_h / _h
    _w = int(_w * _scale)
    _h = int(_h * _scale)

    grid_side_size = 8
    _x = [int(i + 0.5) * _w / grid_side_size for i in range(grid_side_size)]
    _y = [int(i + 0.5) * _h / grid_side_size for i in range(grid_side_size)]
    input_points = np.meshgrid(_x, _y)
    input_points = np.stack(input_points, axis=-1).reshape(-1, 2)
    input_points = input_points / np.array([_w, _h])  # normalize to [0, 1] range
    input_points = input_points.astype(np.float32)

    # warmup the model by running on a sample image
    _sample_image = np.array(_ref_image)

    mask_generator = EfficientTAMAutomaticMaskGenerator(
        model=efficienttam,
        points_per_batch=64,  # we actually need 25 points (5x5 grid) but this is default
        pred_iou_thresh=0.7,
        stability_score_thresh=0.9,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        point_grids=[input_points, input_points],
        points_per_side=None,
        crop_overlap_ratio=0.25,
    )

    _mask = mask_generator.generate(_sample_image)
    # visualize the masks
    _drawn_image = draw_anns(_mask, borders=True)
    _, _ax = plt.subplots(1, 2, figsize=(10, 5))
    _ax[0].imshow(_sample_image)
    _ax[0].scatter(input_points[:, 0] * _w, input_points[:, 1] * _h, c="red", s=10)
    _ax[0].set_title("Sample Image with Prompt Points")
    _ax[1].imshow(_drawn_image)
    _ax[1].set_title(f"Generated Masks ({len(_mask)})")
    _ax[1].axis("off")
    plt.tight_layout()
    plt.gca()
    return (mask_generator,)


@app.cell
def _(
    F,
    Image,
    data_path,
    device,
    extract_dense_visual_features,
    image_files,
    np,
    os,
    pca_on_image_features,
    plt,
    siglip2_model,
    siglip2_patch_size,
    siglip2_processor,
    torch,
):
    # Extract dense SigLIP2 features and visualize PCA + text similarity
    _image = Image.open(os.path.join(data_path, "rgb", image_files[0])).convert("RGB")
    _orig_w, _orig_h = _image.size

    # Resize longest side to 384, pad to square
    _model_size = siglip2_model.vision_model.config.image_size  # 384
    _scale = _model_size / max(_orig_w, _orig_h)
    _new_w, _new_h = int(_orig_w * _scale), int(_orig_h * _scale)
    _resized = _image.resize((_new_w, _new_h), Image.Resampling.LANCZOS)
    # Pad to square (384x384) with black
    _padded = Image.new("RGB", (_model_size, _model_size), (0, 0, 0))
    _padded.paste(_resized, (0, 0))

    # Process through SigLIP2
    _inputs = siglip2_processor(images=_padded, return_tensors="pt").to(device)
    with torch.no_grad():
        _vision_out = siglip2_model.vision_model(_inputs["pixel_values"])
    _dense_tokens = _vision_out.last_hidden_state  # (1, 729, 1152) — no CLS token

    _image_features = extract_dense_visual_features(
        _dense_tokens, image_size=(_model_size, _model_size), patch_size=siglip2_patch_size
    )  # (1, 1152, 27, 27) interpolated to (1, 1152, 384, 384)

    # Crop out the padded region to get features for the actual image content
    _image_features = _image_features[:, :, :_new_h, :_new_w]

    _pca_features = pca_on_image_features(
        _image_features.cpu().numpy(), n_components=3, normalize=True
    ).transpose(1, 2, 0)  # (H, W, n_components)

    # Prepare display image (resize original to match feature spatial dims)
    _display_image = np.array(_resized)
    _pca_features = (_pca_features * 255).astype(np.uint8)

    _, _ax = plt.subplots(1, 2, figsize=(10, 5))
    _ax[0].imshow(_display_image)
    _ax[0].set_title("SigLIP2 Input Image")
    _ax[0].axis("off")
    _ax[1].imshow(_pca_features)
    _ax[1].set_title("PCA Features (3 components)")
    _ax[1].axis("off")
    plt.tight_layout()
    plt.show()

    # Test encoding some text and computing the similarity with the visual features
    _text_prompts = ["book"]
    _text_inputs = siglip2_processor(
        text=_text_prompts, return_tensors="pt", padding="max_length", max_length=64
    ).to(device)
    # Need a dummy pixel_values for the full forward to get text_embeds
    _dummy_pv = _inputs["pixel_values"][:1]
    with torch.no_grad():
        _full_out = siglip2_model(
            input_ids=_text_inputs["input_ids"], pixel_values=_dummy_pv
        )
    _text_features = _full_out.text_embeds  # (1, 1152) — L2-normalized

    _image_features = F.normalize(_image_features, dim=1)
    assert _text_features.shape[1] == _image_features.shape[1], (
        f"Text features shape {_text_features.shape} does not match image features shape {_image_features.shape}"
    )

    # Compute the similarity map (no negation — SigLIP2 uses sigmoid loss,
    # no CLS-leakage inversion like CLIP)
    _sim_map = torch.einsum("bchw,bc->bhw", _image_features, _text_features)  # (H, W)
    _similarity = _sim_map.squeeze(0).cpu().numpy()
    _similarity = (_similarity - _similarity.min()) / (_similarity.max() - _similarity.min())

    _, _ax = plt.subplots(1, 2, figsize=(10, 5))
    _ax[0].imshow(_display_image)
    _ax[0].set_title("Input Image")
    _ax[0].axis("off")
    _bar = _ax[1].imshow(_similarity, cmap="turbo", vmin=0, vmax=1)
    _ax[1].set_title(f"Similarity with {_text_prompts[0]}")
    plt.colorbar(_bar, ax=_ax[1], fraction=0.046, pad=0.04, shrink=0.8)
    _ax[1].axis("off")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(
    Image,
    camera_P,
    create_posed_cloud,
    cv2,
    data_path,
    depth_files,
    gt_poses,
    image_files,
    np,
    o3d,
    os,
    plt,
    tqdm,
):
    # Build a global point cloud from depth + RGB + GT poses
    origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    _viewer = o3d.visualization.Visualizer()
    _viewer.create_window(width=640, height=480, visible=False)
    _viewer.add_geometry(origin_axes)

    big_cloud = o3d.geometry.PointCloud()
    max_views = 20

    for _i, _depth_image_file in enumerate(
        tqdm(depth_files, total=len(depth_files), desc="Processing depth images")
    ):
        if _i > max_views:
            break
        _depth_image_path = os.path.join(data_path, "depth", _depth_image_file)
        _rgb_image_file = image_files[_i]
        _depth_image = cv2.imread(_depth_image_path, cv2.IMREAD_UNCHANGED) / 5000.0  # Convert to meters
        _color_image_path = os.path.join(data_path, "rgb", _rgb_image_file)
        _color_image = np.array(Image.open(_color_image_path).convert("RGB"))
        if _color_image.shape[:2] != _depth_image.shape[:2]:
            _color_image = cv2.resize(
                _color_image,
                (_depth_image.shape[1], _depth_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        _cam_pose = np.eye(4, dtype=np.float64)
        _cam_pose[:3, :3] = gt_poses["R"][_i]
        _cam_pose[:3, 3] = gt_poses["T"][_i]
        _pcd, _cam_axis = create_posed_cloud(
            _depth_image.astype(np.float32),
            _color_image,
            camera_P[:3, :3],
            camera_pose=_cam_pose,
            vlm_features=None,
            with_visualization=True,
        )

        _voxel_size = 0.01  # 1 cm voxel size
        _pcd = _pcd.voxel_down_sample(_voxel_size)
        big_cloud += _pcd

        _viewer.add_geometry(_cam_axis)
        _viewer.add_geometry(_pcd)
        _viewer.poll_events()
        _viewer.update_renderer()

    # Set viewer camera: reuse the last camera pose, just move back a bit
    _cam_params = o3d.camera.PinholeCameraParameters()
    _cam_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=_depth_image.shape[1],
        height=_depth_image.shape[0],
        fx=camera_P[0, 0],
        fy=camera_P[1, 1],
        cx=camera_P[0, 2],
        cy=camera_P[1, 2],
    )
    _T_back = np.eye(4, dtype=np.float64)
    _T_back[:3, 3] = np.array([0, 0, -2.5])  # move back 2.5 meters
    _cam_pose = _cam_pose @ _T_back
    _cam_params.extrinsic = np.linalg.inv(_cam_pose)
    _viewer.get_view_control().convert_from_pinhole_camera_parameters(_cam_params, allow_arbitrary=True)
    _viewer.update_renderer()

    _screenshot = np.asarray(_viewer.capture_screen_float_buffer(do_render=True))
    plt.figure(figsize=(10, 10))
    plt.imshow(_screenshot)
    plt.axis("off")
    plt.show()

    _viewer.destroy_window()
    return (big_cloud, origin_axes)


@app.cell
def _(
    Image,
    big_cloud,
    camera_P,
    cm,
    cv2,
    data_path,
    depth_files,
    depth_to_point_cloud,
    device,
    extract_dense_visual_features,
    gt_poses,
    image_files,
    mask_generator,
    np,
    o3d,
    os,
    siglip2_model,
    siglip2_patch_size,
    siglip2_processor,
    torch,
    tqdm,
):
    # For demo purpose, use a uniform voxel grid to fuse/merge pc features
    # This would be where we use supereight or other mapping tools
    voxel_size = [0.03, 0.03, 0.03]  # 3 cm voxel size
    _cloud_dims = np.array(big_cloud.get_max_bound()) - np.array(big_cloud.get_min_bound())
    _num_voxels = np.ceil(_cloud_dims / voxel_size).astype(int)
    print("Number of voxels:", _num_voxels)

    _x_coords = np.linspace(big_cloud.get_min_bound()[0], big_cloud.get_max_bound()[0], _num_voxels[0])
    _y_coords = np.linspace(big_cloud.get_min_bound()[1], big_cloud.get_max_bound()[1], _num_voxels[1])
    _z_coords = np.linspace(big_cloud.get_min_bound()[2], big_cloud.get_max_bound()[2], _num_voxels[2])
    voxel_centers = np.array(
        np.meshgrid(_x_coords, _y_coords, _z_coords, indexing="ij")
    ).reshape(3, -1).T  # (N, 3)

    _voxel_centers_pcd = o3d.geometry.PointCloud()
    _voxel_centers_pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    _kd_tree = o3d.geometry.KDTreeFlann(_voxel_centers_pcd)
    vlm_world_features = {}  # voxel_id -> avg VLM features

    _model_size = siglip2_model.vision_model.config.image_size  # 384

    for _i in tqdm(range(20)):
        _rgb_image_file = image_files[_i]
        _depth_image_file = depth_files[_i]
        _depth_image_path = os.path.join(data_path, "depth", _depth_image_file)
        _rgb_image_path = os.path.join(data_path, "rgb", _rgb_image_file)
        _depth_image = cv2.imread(_depth_image_path, cv2.IMREAD_UNCHANGED) / 5000.0
        _rgb_image = np.array(Image.open(_rgb_image_path).convert("RGB"))
        if _rgb_image.shape[:2] != _depth_image.shape[:2]:
            _rgb_image = cv2.resize(
                _rgb_image,
                (_depth_image.shape[1], _depth_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # run the mask generator on the RGB image
        _masks = mask_generator.generate(_rgb_image)

        # Extract SigLIP2 features: resize longest to model_size, pad to square
        _h, _w = _rgb_image.shape[:2]
        _scale = _model_size / max(_h, _w)
        _new_h, _new_w = int(_h * _scale), int(_w * _scale)
        _resized = cv2.resize(_rgb_image, (_new_w, _new_h), interpolation=cv2.INTER_LINEAR)
        _padded = np.zeros((_model_size, _model_size, 3), dtype=np.uint8)
        _padded[:_new_h, :_new_w] = _resized

        _inputs = siglip2_processor(images=Image.fromarray(_padded), return_tensors="pt").to(device)
        with torch.no_grad():
            _vision_out = siglip2_model.vision_model(_inputs["pixel_values"])
        _dense_tokens = _vision_out.last_hidden_state  # (1, 729, 1152)
        _dense_features = extract_dense_visual_features(
            _dense_tokens, (_model_size, _model_size), patch_size=siglip2_patch_size
        )  # (1, C, 384, 384)

        # Crop out padding to get features for actual image content
        _dense_features = _dense_features[:, :, :_new_h, :_new_w]
        _dense_features /= _dense_features.norm(dim=1, keepdim=True)

        _numpy_vlm_features = _dense_features.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        _numpy_vlm_features = (
            (_numpy_vlm_features - _numpy_vlm_features.min())
            / (_numpy_vlm_features.max() - _numpy_vlm_features.min())
        )

        # filter out regions not covered by masks
        _mask = np.zeros(_rgb_image.shape[:2], dtype=np.uint8)
        for _m in _masks:
            _mask[_m["segmentation"]] = 1
        _depth_image[_mask == 0] = 0.0

        # downsample rgb and depth to match VLM features resolution
        _fh, _fw = _numpy_vlm_features.shape[:2]
        _scale_x = _fw / _rgb_image.shape[1]
        _scale_y = _fh / _rgb_image.shape[0]
        _rgb_image = cv2.resize(_rgb_image, (_fw, _fh), interpolation=cv2.INTER_LINEAR)
        _depth_image = cv2.resize(_depth_image, (_fw, _fh), interpolation=cv2.INTER_NEAREST)

        # scale intrinsics to match new image size
        _P_scaled = camera_P.copy()
        _P_scaled[0, 0] *= _scale_x  # fx
        _P_scaled[1, 1] *= _scale_y  # fy
        _P_scaled[0, 2] *= _scale_x  # cx
        _P_scaled[1, 2] *= _scale_y  # cy

        _pose = np.eye(4, dtype=np.float64)
        _pose[:3, :3] = gt_poses["R"][_i]
        _pose[:3, 3] = gt_poses["T"][_i]
        _point_cloud_numpy = depth_to_point_cloud(
            _depth_image.astype(np.float32),
            _rgb_image,
            _P_scaled[:3, :3],
            vlm_features=_numpy_vlm_features,
        )

        # assign point cloud to voxel centers via KD-tree
        for _point in _point_cloud_numpy:
            _point = np.array(_point)
            _point_world = np.dot(_pose[:3, :3], _point[:3]) + _pose[:3, 3]
            _point[:3] = _point_world
            [_k, _idx, _dist] = _kd_tree.search_knn_vector_3d(_point[:3], 1)
            if _k > 0 and _dist[0] < voxel_size[0]:
                if _idx[0] in vlm_world_features:
                    _existing = vlm_world_features[_idx[0]]
                    _new_features = _point[6:]
                    vlm_world_features[_idx[0]]["count"] += 1
                    vlm_world_features[_idx[0]]["features"] = (
                        _existing["features"] * (_existing["count"] - 1) + _new_features
                    ) / _existing["count"]
                else:
                    vlm_world_features[_idx[0]] = {"count": 1, "features": _point[6:]}

    activated_voxels = np.array(list(vlm_world_features.keys()))
    _centers_3d = voxel_centers[activated_voxels]
    activated_cloud = o3d.geometry.PointCloud()
    activated_cloud.points = o3d.utility.Vector3dVector(_centers_3d)

    # color based on observation counts
    _counts = np.array([vlm_world_features[i]["count"] for i in activated_voxels])
    _counts = (_counts - _counts.min()) / (_counts.max() - _counts.min())
    _colors = cm.viridis(_counts)[:, :3]
    activated_cloud.colors = o3d.utility.Vector3dVector(_colors)

    # camera visualization for first pose
    _T = np.eye(4, dtype=np.float64)
    _T[:3, :3] = gt_poses["R"][0]
    _T[:3, 3] = gt_poses["T"][0]
    _extrinsic_matrix = np.linalg.inv(_T)
    cam_axis_viz = o3d.geometry.LineSet.create_camera_visualization(
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=_depth_image.shape[1],
            height=_depth_image.shape[0],
            fx=_P_scaled[0, 0],
            fy=_P_scaled[1, 1],
            cx=_P_scaled[0, 2],
            cy=_P_scaled[1, 2],
        ),
        extrinsic=_extrinsic_matrix,
        scale=0.1,
    )
    return (activated_cloud, activated_voxels, cam_axis_viz, vlm_world_features)


@app.cell
def _(
    F,
    activated_cloud,
    activated_voxels,
    big_cloud,
    cam_axis_viz,
    cm,
    device,
    np,
    o3d,
    origin_axes,
    plt,
    siglip2_model,
    siglip2_processor,
    torch,
    vlm_world_features,
):
    from sklearn.metrics.pairwise import cosine_similarity

    # Query the voxel map with a text prompt
    _voxel_features = np.array([vlm_world_features[i]["features"] for i in activated_voxels])

    text_query = "a white keyboard"
    _text_inputs = siglip2_processor(
        text=[text_query], return_tensors="pt", padding="max_length", max_length=64
    ).to(device)
    # Use a dummy pixel_values for the full forward to get text_embeds
    from PIL import Image as _Image
    _dummy_inputs = siglip2_processor(images=_Image.new("RGB", (384, 384)), return_tensors="pt").to(device)
    with torch.no_grad():
        _full_out = siglip2_model(
            input_ids=_text_inputs["input_ids"], pixel_values=_dummy_inputs["pixel_values"]
        )
    _text_features = F.normalize(_full_out.text_embeds, dim=-1).cpu().numpy()

    _similarity_scores = cosine_similarity(_voxel_features, _text_features).flatten()
    _similarity_scores = (
        (_similarity_scores - _similarity_scores.min())
        / (_similarity_scores.max() - _similarity_scores.min())
    )
    _similarity_colors = cm.viridis(_similarity_scores)[:, :3]
    activated_cloud.colors = o3d.utility.Vector3dVector(_similarity_colors)

    # Visualize similarity cloud
    _viewer = o3d.visualization.Visualizer()
    _viewer.create_window(width=640, height=480, visible=False)
    _viewer.add_geometry(activated_cloud)
    _viewer.add_geometry(origin_axes)
    _viewer.add_geometry(cam_axis_viz)
    _viewer.poll_events()
    _viewer.update_renderer()
    _viewer.get_view_control().set_zoom(0.3)

    _screenshot_sim = np.asarray(_viewer.capture_screen_float_buffer(do_render=True))

    # Switch to RGB point cloud for comparison
    _viewer.clear_geometries()
    _viewer.add_geometry(big_cloud)
    _viewer.add_geometry(origin_axes)
    _viewer.add_geometry(cam_axis_viz)
    _viewer.poll_events()
    _viewer.update_renderer()
    _viewer.get_view_control().set_zoom(0.3)

    _screenshot_rgb = np.asarray(_viewer.capture_screen_float_buffer(do_render=True))

    _, _ax = plt.subplots(1, 2, figsize=(10, 5))
    _ax[0].imshow(_screenshot_sim)
    _ax[0].set_title(f"SigLIP2 Features - Similarity to Query\n'{text_query}'")
    _ax[0].axis("off")
    _ax[1].imshow(_screenshot_rgb)
    _ax[1].set_title("RGB Point Cloud")
    _ax[1].axis("off")
    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
