from huggingface_hub import scan_cache_dir


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
        "FastSAM-x.pt",
        "FastSAM-s.pt",
        ]
    return sam_model_ids


inp_list_from_cache = None


def get_inp_model_ids():
    """Get inpainting model ids list.

    Returns:
        list: model ids list
    """
    global inp_list_from_cache
    model_ids = [
        "stabilityai/stable-diffusion-2-inpainting",
        "Uminosachi/dreamshaper_6Inpainting",
        "Uminosachi/Deliberate-inpainting",
        "Uminosachi/realisticVisionV30_v30VAE-inpainting",
        "Uminosachi/revAnimated_v121Inp-inpainting",
        "runwayml/stable-diffusion-inpainting",
        ]
    if inp_list_from_cache is not None and isinstance(inp_list_from_cache, list):
        model_ids.extend(inp_list_from_cache)
        return model_ids
    try:
        hf_cache_info = scan_cache_dir()
        inpaint_repos = []
        for repo in hf_cache_info.repos:
            if repo.repo_type == "model" and "inpaint" in repo.repo_id.lower() and repo.repo_id not in model_ids:
                inpaint_repos.append(repo.repo_id)
        inp_list_from_cache = sorted(inpaint_repos, reverse=True, key=lambda x: x.split("/")[-1])
        model_ids.extend(inp_list_from_cache)
        return model_ids
    except Exception:
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
