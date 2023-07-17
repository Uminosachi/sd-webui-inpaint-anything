from .fast_sam_wrapper import FastSAM
from .fast_sam_wrapper import FastSamAutomaticMaskGenerator

fast_sam_model_registry = {
    "FastSAM-x": FastSAM,
    "FastSAM-s": FastSAM,
}

__all__ = ["FastSAM", "FastSamAutomaticMaskGenerator", "fast_sam_model_registry"]
