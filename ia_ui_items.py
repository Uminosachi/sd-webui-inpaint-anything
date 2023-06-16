def get_sampler_names():
    """Get sampler name list.

    Returns:
        list: sampler name list
    """
    sampler_names = [
        "DDIM",
        "Euler",
        "Euler a",
        "DPM2 Karras",
        "DPM2 a Karras",
        ]
    return sampler_names

def get_sam_model_ids():
    """Get SAM model ids list.

    Returns:
        list: SAM model ids list
    """
    sam_model_ids = [
        "sam_vit_h_4b8939.pth",
        "sam_vit_l_0b3195.pth",
        "sam_vit_b_01ec64.pth",
        "sam_hq_vit_h.pth",
        "sam_hq_vit_l.pth",
        "sam_hq_vit_b.pth",
        ]
    return sam_model_ids

def get_model_ids():
    """Get inpainting model ids list.

    Returns:
        list: model ids list
    """
    model_ids = [
        "stabilityai/stable-diffusion-2-inpainting",
        "Uminosachi/dreamshaper_6Inpainting",
        "Uminosachi/dreamshaper_5-inpainting",
        "Uminosachi/Deliberate-inpainting",
        "saik0s/realistic_vision_inpainting",
        "Uminosachi/revAnimated_v121Inp-inpainting",
        "parlance/dreamlike-diffusion-1.0-inpainting",
        "runwayml/stable-diffusion-inpainting",
        ]
    return model_ids

def get_cleaner_model_ids():
    """Get cleaner model ids list.

    Returns:
        list: model ids list
    """
    model_ids = [
        "lama",
        "ldm",
        "zits",
        "mat",
        "fcf",
        "manga",
        ]
    return model_ids

def get_padding_mode_names():
    """Get padding mode name list.
    
    Returns:
        list: padding mode name list
    """
    padding_mode_names = [
        "constant",
        "edge",
        "reflect",
        "mean",
        "median",
        "maximum",
        "minimum",
        ]
    return padding_mode_names
