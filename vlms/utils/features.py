import numpy as np
import torch
from sklearn.decomposition import PCA


def extract_clip_image_features(
    clip_features: torch.Tensor, image_size: tuple
) -> torch.Tensor:

    clip_features = clip_features[:, 1:, :]  # remove the CLS token
    if len(clip_features.shape) != 3:
        raise ValueError(
            f"Expected clip_features to be of shape (B, N, C), got {clip_features.shape}"
        )
    if clip_features.shape[0] != 1:
        raise ValueError(f"Expected batch size of 1, got {clip_features.shape[0]}")
    if len(image_size) != 2:
        raise ValueError(
            f"Expected image_size to be of length 2, got {len(image_size)}"
        )
    h, w = image_size
    n_side_patches = h // 14  # hardcoded patch size of 14x14 used in the model
    clip_features = clip_features.reshape(
        1, n_side_patches, n_side_patches, -1
    ).permute(
        0, 3, 1, 2
    )  # (B, C, H, W)
    clip_features /= clip_features.norm(
        dim=1, keepdim=True
    )  # normalize features along the channel dimension
    image_features = torch.nn.functional.interpolate(
        clip_features, size=(h, w), mode="bilinear", align_corners=False
    )
    return image_features


def pca_on_image_features(
    image_features: np.ndarray, n_components: int = 3, normalize: bool = False
) -> np.ndarray:

    # we expect image_features to be of shape (B, C, H, W)
    if len(image_features.shape) != 4:
        raise ValueError(
            f"Expected image_features to be of shape (B, C, H, W), got {image_features.shape}"
        )
    # also assume/expect BS=1
    if image_features.shape[0] != 1:
        raise ValueError(f"Expected batch size of 1, got {image_features.shape[0]}")
    image_features = image_features.squeeze(0)  # (C, H, W)
    C, H, W = image_features.shape
    if C < n_components:
        raise ValueError(
            f"Number of components {n_components} cannot be greater than number of channels {C}"
        )

    image_features = image_features.reshape(C, -1)  # (C, H*W)
    image_features = image_features.T  # (H*W, C)
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(image_features)  # (H*W, n_components)
    pca_features = pca_features.reshape(H, W, n_components)
    if normalize:
        # min-max normalize the PCA features to [0, 1]
        pca_features = (pca_features - pca_features.min()) / (
            pca_features.max() - pca_features.min()
        )
    return pca_features.transpose(2, 0, 1)  # (n_components, H, W)
