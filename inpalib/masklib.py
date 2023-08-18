from typing import Any

import numpy as np


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Invert mask.

    Args:
        mask (np.ndarray): mask

    Returns:
        np.ndarray: inverted mask
    """
    if mask is None or type(mask) != np.ndarray:
        raise ValueError("Invalid mask")

    return np.logical_not(mask.astype(bool)).astype(np.uint8) * 255


def check_inputs_create_mask_image(
        mask: np.ndarray,
        sam_masks: list[dict[str, Any]],
        ignore_black_chk: bool = True,
        ) -> None:
    """Check create mask image inputs.

    Args:
        mask (np.ndarray): mask
        sam_masks (list[dict[str, Any]]): SAM masks
        ignore_black_chk (bool): ignore black check

    Returns:
        None
    """
    if mask is None or type(mask) != np.ndarray:
        raise ValueError("Invalid mask")
    elif mask.ndim != 3 or (mask.shape[2] != 1 and mask.shape[2] != 3):
        raise ValueError("Mask must be 3 dimensional with 1 or 3 channels")

    if sam_masks is None or type(sam_masks) != list:
        raise ValueError("Invalid SAM masks")

    if ignore_black_chk is None or type(ignore_black_chk) != bool:
        raise ValueError("Invalid ignore black check")


def create_mask_image(
        mask: np.ndarray,
        sam_masks: list[dict[str, Any]],
        ignore_black_chk: bool = True,
        ) -> np.ndarray:
    """Create mask image.

    Args:
        mask (np.ndarray): mask
        sam_masks (list[dict[str, Any]]): SAM masks
        ignore_black_chk (bool): ignore black check

    Returns:
        np.ndarray: mask image
    """
    check_inputs_create_mask_image(mask, sam_masks, ignore_black_chk)
    if mask.shape[2] == 3:
        mask = mask[:, :, 0:1]

    if len(sam_masks) > 0 and sam_masks[0]["segmentation"].shape[:2] != mask.shape[:2]:
        raise ValueError("sam_masks shape not match")

    canvas_image = np.zeros(mask.shape, dtype=np.uint8)
    mask_region = np.zeros(mask.shape, dtype=np.uint8)
    for idx, seg_dict in enumerate(sam_masks):
        seg_mask = np.expand_dims(seg_dict["segmentation"].astype(np.uint8), axis=-1)
        canvas_mask = np.logical_not(canvas_image.astype(bool)).astype(np.uint8)
        if (seg_mask * canvas_mask * mask).astype(bool).any():
            mask_region = mask_region + (seg_mask * canvas_mask)
        seg_color = seg_mask * canvas_mask
        canvas_image = canvas_image + seg_color

    if not ignore_black_chk:
        canvas_mask = np.logical_not(canvas_image.astype(bool)).astype(np.uint8)
        if (canvas_mask * mask).astype(bool).any():
            mask_region = mask_region + (canvas_mask)

    mask_region = np.tile(mask_region * 255, (1, 1, 3))

    seg_image = mask_region.astype(np.uint8)

    return seg_image
