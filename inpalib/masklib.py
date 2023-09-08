from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Invert mask.

    Args:
        mask (np.ndarray): mask

    Returns:
        np.ndarray: inverted mask
    """
    if mask is None or not isinstance(mask, np.ndarray):
        raise ValueError("Invalid mask")

    # return np.logical_not(mask.astype(bool)).astype(np.uint8) * 255
    return np.invert(mask.astype(np.uint8))


def check_inputs_create_mask_image(
        mask: Union[np.ndarray, Image.Image],
        sam_masks: List[Dict[str, Any]],
        ignore_black_chk: bool = True,
        ) -> None:
    """Check create mask image inputs.

    Args:
        mask (Union[np.ndarray, Image.Image]): mask
        sam_masks (List[Dict[str, Any]]): SAM masks
        ignore_black_chk (bool): ignore black check

    Returns:
        None
    """
    if mask is None or not isinstance(mask, (np.ndarray, Image.Image)):
        raise ValueError("Invalid mask")

    if sam_masks is None or not isinstance(sam_masks, list):
        raise ValueError("Invalid SAM masks")

    if ignore_black_chk is None or not isinstance(ignore_black_chk, bool):
        raise ValueError("Invalid ignore black check")


def convert_mask(mask: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Convert mask.

    Args:
        mask (Union[np.ndarray, Image.Image]): mask

    Returns:
        np.ndarray: converted mask
    """
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    if mask.shape[2] != 1:
        mask = mask[:, :, 0:1]

    return mask


def create_mask_image(
        mask: Union[np.ndarray, Image.Image],
        sam_masks: List[Dict[str, Any]],
        ignore_black_chk: bool = True,
        ) -> np.ndarray:
    """Create mask image.

    Args:
        mask (Union[np.ndarray, Image.Image]): mask
        sam_masks (List[Dict[str, Any]]): SAM masks
        ignore_black_chk (bool): ignore black check

    Returns:
        np.ndarray: mask image
    """
    check_inputs_create_mask_image(mask, sam_masks, ignore_black_chk)
    mask = convert_mask(mask)

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
