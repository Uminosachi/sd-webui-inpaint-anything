import os
import platform

import torch
from modules import devices

from fast_sam import FastSamAutomaticMaskGenerator, fast_sam_model_registry
from ia_check_versions import ia_check_versions
from ia_config import get_webui_setting
from ia_logging import ia_logging
from ia_threading import torch_default_load_cd
from mobile_sam import SamAutomaticMaskGenerator as SamAutomaticMaskGeneratorMobile
from mobile_sam import SamPredictor as SamPredictorMobile
from mobile_sam import sam_model_registry as sam_model_registry_mobile
from segment_anything_fb import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from segment_anything_hq import SamAutomaticMaskGenerator as SamAutomaticMaskGeneratorHQ
from segment_anything_hq import SamPredictor as SamPredictorHQ
from segment_anything_hq import sam_model_registry as sam_model_registry_hq


@torch_default_load_cd()
def get_sam_mask_generator(sam_checkpoint, anime_style_chk=False):
    """Get SAM mask generator.

    Args:
        sam_checkpoint (str): SAM checkpoint path

    Returns:
        SamAutomaticMaskGenerator or None: SAM mask generator
    """
    # model_type = "vit_h"
    if "_hq_" in os.path.basename(sam_checkpoint):
        model_type = os.path.basename(sam_checkpoint)[7:12]
        sam_model_registry_local = sam_model_registry_hq
        SamAutomaticMaskGeneratorLocal = SamAutomaticMaskGeneratorHQ
        points_per_batch = 32
    elif "FastSAM" in os.path.basename(sam_checkpoint):
        model_type = os.path.splitext(os.path.basename(sam_checkpoint))[0]
        sam_model_registry_local = fast_sam_model_registry
        SamAutomaticMaskGeneratorLocal = FastSamAutomaticMaskGenerator
        points_per_batch = None
    elif "mobile_sam" in os.path.basename(sam_checkpoint):
        model_type = "vit_t"
        sam_model_registry_local = sam_model_registry_mobile
        SamAutomaticMaskGeneratorLocal = SamAutomaticMaskGeneratorMobile
        points_per_batch = 64
    else:
        model_type = os.path.basename(sam_checkpoint)[4:9]
        sam_model_registry_local = sam_model_registry
        SamAutomaticMaskGeneratorLocal = SamAutomaticMaskGenerator
        points_per_batch = 64

    pred_iou_thresh = 0.88 if not anime_style_chk else 0.83
    stability_score_thresh = 0.95 if not anime_style_chk else 0.9

    if os.path.isfile(sam_checkpoint):
        sam = sam_model_registry_local[model_type](checkpoint=sam_checkpoint)
        if platform.system() == "Darwin":
            if "FastSAM" in os.path.basename(sam_checkpoint) or not ia_check_versions.torch_mps_is_available:
                sam.to(device=torch.device("cpu"))
            else:
                sam.to(device=torch.device("mps"))
        else:
            if get_webui_setting("inpaint_anything_sam_oncpu", False):
                ia_logging.info("SAM is running on CPU... (the option has been checked)")
                sam.to(device=devices.cpu)
            else:
                sam.to(device=devices.device)
        sam_mask_generator = SamAutomaticMaskGeneratorLocal(
            model=sam, points_per_batch=points_per_batch, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh)
    else:
        sam_mask_generator = None

    return sam_mask_generator


@torch_default_load_cd()
def get_sam_predictor(sam_checkpoint):
    """Get SAM predictor.

    Args:
        sam_checkpoint (str): SAM checkpoint path

    Returns:
        SamPredictor or None: SAM predictor
    """
    # model_type = "vit_h"
    if "_hq_" in os.path.basename(sam_checkpoint):
        model_type = os.path.basename(sam_checkpoint)[7:12]
        sam_model_registry_local = sam_model_registry_hq
        SamPredictorLocal = SamPredictorHQ
    elif "FastSAM" in os.path.basename(sam_checkpoint):
        raise NotImplementedError("FastSAM predictor is not implemented yet.")
    elif "mobile_sam" in os.path.basename(sam_checkpoint):
        model_type = "vit_t"
        sam_model_registry_local = sam_model_registry_mobile
        SamPredictorLocal = SamPredictorMobile
    else:
        model_type = os.path.basename(sam_checkpoint)[4:9]
        sam_model_registry_local = sam_model_registry
        SamPredictorLocal = SamPredictor

    if os.path.isfile(sam_checkpoint):
        sam = sam_model_registry_local[model_type](checkpoint=sam_checkpoint)
        if platform.system() == "Darwin":
            if "FastSAM" in os.path.basename(sam_checkpoint) or not ia_check_versions.torch_mps_is_available:
                sam.to(device=torch.device("cpu"))
            else:
                sam.to(device=torch.device("mps"))
        else:
            if get_webui_setting("inpaint_anything_sam_oncpu", False):
                ia_logging.info("SAM is running on CPU... (the option has been checked)")
                sam.to(device=devices.cpu)
            else:
                sam.to(device=devices.device)
        sam_predictor = SamPredictorLocal(sam)
    else:
        sam_predictor = None

    return sam_predictor
