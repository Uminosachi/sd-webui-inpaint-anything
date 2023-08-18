import os
import traceback
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from ia_file_manager import ia_file_manager
from ia_get_dataset_colormap import create_pascal_label_colormap
from ia_logging import ia_logging
from ia_sam_manager import get_sam_mask_generator
from ia_ui_items import get_sam_model_ids


def get_all_sam_ids() -> list[str]:
    """Get all SAM IDs.

    Returns:
        list[str]: SAM IDs
    """
    return get_sam_model_ids()


def sam_file_path(sam_id: str) -> str:
    """Get SAM file path.

    Args:
        sam_id (str): SAM ID

    Returns:
        str: SAM file path
    """
    return os.path.join(ia_file_manager.models_dir, sam_id)


def sam_file_exists(sam_id: str) -> bool:
    """Check if SAM file exists.

    Args:
        sam_id (str): SAM ID

    Returns:
        bool: True if SAM file exists else False
    """
    sam_checkpoint = sam_file_path(sam_id)

    return os.path.isfile(sam_checkpoint)


def get_available_sam_ids() -> list[str]:
    """Get available SAM IDs.

    Returns:
        list[str]: available SAM IDs
    """
    all_sam_ids = get_all_sam_ids()
    for sam_id in all_sam_ids.copy():
        if not sam_file_exists(sam_id):
            all_sam_ids.remove(sam_id)

    return all_sam_ids


def check_inputs_generate_sam_mask(
        input_image: np.ndarray,
        sam_id: str,
        anime_style_chk: bool = False,
        ) -> None:
    """Check generate SAM mask inputs.

    Args:
        input_image (np.ndarray): input image
        sam_id (str): SAM ID
        anime_style_chk (bool): anime style check

    Returns:
        None
    """
    if input_image is None or type(input_image) != np.ndarray:
        raise ValueError("Invalid input image")
    elif input_image.ndim != 3 or input_image.shape[2] != 3:
        raise ValueError("Input image must be 3 dimensional with 3 channels")

    if sam_id is None or type(sam_id) != str:
        raise ValueError("Invalid SAM ID")
    elif sam_id not in get_available_sam_ids():
        raise ValueError(f"SAM ID {sam_id} not available")

    if anime_style_chk is None or type(anime_style_chk) != bool:
        raise ValueError("Invalid anime style check")


def generate_sam_mask(
        input_image: np.ndarray,
        sam_id: str,
        anime_style_chk: bool = False,
        ) -> np.ndarray:
    """Generate SAM mask.

    Args:
        input_image (np.ndarray): input image
        sam_id (str): SAM ID
        anime_style_chk (bool): anime style check

    Returns:
        np.ndarray: SAM mask
    """
    try:
        check_inputs_generate_sam_mask(input_image, sam_id, anime_style_chk)
    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        return None

    sam_checkpoint = sam_file_path(sam_id)
    sam_mask_generator = get_sam_mask_generator(sam_checkpoint, anime_style_chk)
    ia_logging.info(f"{sam_mask_generator.__class__.__name__} {sam_id}")
    try:
        sam_masks = sam_mask_generator.generate(input_image)
    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        del sam_mask_generator
        return None

    if anime_style_chk:
        for sam_mask in sam_masks:
            sam_mask_seg = sam_mask["segmentation"]
            sam_mask_seg = cv2.morphologyEx(sam_mask_seg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            sam_mask_seg = cv2.morphologyEx(sam_mask_seg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            sam_mask["segmentation"] = sam_mask_seg.astype(bool)

    ia_logging.info("sam_masks: {}".format(len(sam_masks)))

    return sam_masks


def sort_mask_by_size(sam_masks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort mask by size.

    Args:
        sam_masks (list[dict[str, Any]]): SAM masks

    Returns:
        list[dict[str, Any]]: sorted SAM masks
    """
    return sorted(sam_masks, key=lambda x: np.sum(x.get("segmentation").astype(np.uint32)))


def get_seg_colormap() -> np.ndarray:
    """Get segmentation colormap.

    Returns:
        np.ndarray: segmentation colormap
    """
    cm_pascal = create_pascal_label_colormap()
    seg_colormap = cm_pascal
    seg_colormap = np.array([c for c in seg_colormap if max(c) >= 64], dtype=np.uint8)

    return seg_colormap


def insert_mask_to_sam_masks(
        sam_masks: list[dict[str, Any]],
        insert_mask: dict[str, Any],
        ) -> list[dict[str, Any]]:
    """Insert mask to SAM masks.

    Args:
        sam_masks (list[dict[str, Any]]): SAM masks
        insert_mask (dict[str, Any]): insert mask

    Returns:
        list[dict[str, Any]]: SAM masks
    """
    if insert_mask is not None and type(insert_mask) == dict and "segmentation" in insert_mask:
        if (len(sam_masks) > 0 and
                sam_masks[0]["segmentation"].shape == insert_mask["segmentation"].shape and
                np.any(insert_mask["segmentation"])):
            sam_masks.insert(0, insert_mask)
            ia_logging.info("insert pad_mask to sam_masks")

    return sam_masks


def create_seg_color_image(
        input_image: np.ndarray,
        sam_masks: list[dict[str, Any]],
        ) -> np.ndarray:
    """Create segmentation color image.

    Args:
        input_image (np.ndarray): input image
        sam_masks (list[dict[str, Any]]): SAM masks

    Returns:
        np.ndarray: segmentation color image
    """
    seg_colormap = get_seg_colormap()
    sam_masks = sam_masks[:len(seg_colormap)]

    with tqdm(total=len(sam_masks), desc="Processing segments") as progress_bar:
        canvas_image = np.zeros((*input_image.shape[:2], 1), dtype=np.uint8)
        for idx, seg_dict in enumerate(sam_masks[0:min(255, len(sam_masks))]):
            seg_mask = np.expand_dims(seg_dict["segmentation"].astype(np.uint8), axis=-1)
            canvas_mask = np.logical_not(canvas_image.astype(bool)).astype(np.uint8)
            seg_color = np.array([idx+1], dtype=np.uint8) * seg_mask * canvas_mask
            canvas_image = canvas_image + seg_color
            progress_bar.update(1)
        seg_colormap = np.insert(seg_colormap, 0, [0, 0, 0], axis=0)
        temp_canvas_image = np.apply_along_axis(lambda x: seg_colormap[x[0]], axis=-1, arr=canvas_image)
        if len(sam_masks) > 255:
            canvas_image = canvas_image.astype(bool).astype(np.uint8)
            for idx, seg_dict in enumerate(sam_masks[255:min(509, len(sam_masks))]):
                seg_mask = np.expand_dims(seg_dict["segmentation"].astype(np.uint8), axis=-1)
                canvas_mask = np.logical_not(canvas_image.astype(bool)).astype(np.uint8)
                seg_color = np.array([idx+2], dtype=np.uint8) * seg_mask * canvas_mask
                canvas_image = canvas_image + seg_color
                progress_bar.update(1)
            seg_colormap = seg_colormap[256:]
            seg_colormap = np.insert(seg_colormap, 0, [0, 0, 0], axis=0)
            seg_colormap = np.insert(seg_colormap, 0, [0, 0, 0], axis=0)
            canvas_image = np.apply_along_axis(lambda x: seg_colormap[x[0]], axis=-1, arr=canvas_image)
            canvas_image = temp_canvas_image + canvas_image
        else:
            canvas_image = temp_canvas_image
    ret_seg_image = canvas_image.astype(np.uint8)

    return ret_seg_image
