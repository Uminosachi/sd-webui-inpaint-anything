import os
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from get_dataset_colormap import create_pascal_label_colormap
from torch.hub import download_url_to_file
from torchvision import transforms
from datetime import datetime
import gc
import argparse
import platform
from PIL.PngImagePlugin import PngInfo
import time
import random
import cv2
from huggingface_hub import snapshot_download
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler
# print("platform:", platform.system())

import modules.scripts as scripts
from modules import shared, script_callbacks
try:
    from modules.paths_internal import extensions_dir
except Exception:
    from modules.extensions import extensions_dir
from modules.devices import device, torch_gc
from modules.safe import unsafe_torch_load, load

import re
from webui_controlnet import (find_controlnet, get_sd_img2img_processing,
                              backup_alwayson_scripts, disable_alwayson_scripts, restore_alwayson_scripts,
                              get_controlnet_args_to, clear_controlnet_cache)
from modules.processing import process_images, create_infotext
from modules.sd_samplers import samplers_for_img2img
from modules.images import resize_image
from modules.sd_models import unload_model_weights, reload_model_weights
from segment_anything_hq import sam_model_registry as sam_model_registry_hq
from segment_anything_hq import SamAutomaticMaskGenerator as SamAutomaticMaskGeneratorHQ
from segment_anything_hq import SamPredictor as SamPredictorHQ
from ia_logging import ia_logging
from ia_ui_items import (get_sampler_names, get_sam_model_ids, get_model_ids, get_cleaner_model_ids, get_padding_mode_names)

_DOWNLOAD_COMPLETE = "Download complete"

def download_model(sam_model_id):
    """Download SAM model.

    Args:
        sam_model_id (str): SAM model id

    Returns:
        str: download status
    """
    # print(sam_model_id)
    if "_hq_" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/sam-hq/resolve/main/" + sam_model_id
    else:
        # url_sam_vit_h_4b8939 = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        url_sam = "https://dl.fbaipublicfiles.com/segment_anything/" + sam_model_id
    models_dir = os.path.join(extensions_dir, "sd-webui-inpaint-anything", "models")
    sam_checkpoint = os.path.join(models_dir, sam_model_id)
    if not os.path.isfile(sam_checkpoint):
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        
        download_url_to_file(url_sam, sam_checkpoint)
        
        return _DOWNLOAD_COMPLETE
    else:
        return "Model already exists"

def download_model_from_hf(hf_model_id, local_files_only=False):
    """Download model from HuggingFace Hub.

    Args:
        sam_model_id (str): HuggingFace model id
        local_files_only (bool, optional): If True, use only local files. Defaults to False.

    Returns:
        str: download status
    """
    if not local_files_only:
        ia_logging.info(f"Downloading {hf_model_id}")
    try:
        snapshot_download(repo_id=hf_model_id, local_files_only=local_files_only)
    except FileNotFoundError:
        return f"{hf_model_id} not found, please download"
    except Exception as e:
        return str(e)

    return _DOWNLOAD_COMPLETE

def get_sam_mask_generator(sam_checkpoint):
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
    else:
        model_type = os.path.basename(sam_checkpoint)[4:9]
        sam_model_registry_local = sam_model_registry
        SamAutomaticMaskGeneratorLocal = SamAutomaticMaskGenerator
        points_per_batch = 64

    if os.path.isfile(sam_checkpoint):
        torch.load = unsafe_torch_load
        sam = sam_model_registry_local[model_type](checkpoint=sam_checkpoint)
        if platform.system() == "Darwin":
            sam.to(device="cpu")
        else:
            sam.to(device=device)
        sam_mask_generator = SamAutomaticMaskGeneratorLocal(sam, points_per_batch=points_per_batch)
        torch.load = load
    else:
        sam_mask_generator = None
    
    return sam_mask_generator

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
    else:
        model_type = os.path.basename(sam_checkpoint)[4:9]
        sam_model_registry_local = sam_model_registry
        SamPredictorLocal = SamPredictor

    if os.path.isfile(sam_checkpoint):
        torch.load = unsafe_torch_load
        sam = sam_model_registry_local[model_type](checkpoint=sam_checkpoint)
        if platform.system() == "Darwin":
            sam.to(device="cpu")
        else:
            sam.to(device=device)
        sam_predictor = SamPredictorLocal(sam)
        torch.load = load
    else:
        sam_predictor = None
    
    return sam_predictor

ia_outputs_dir = os.path.join(os.path.dirname(extensions_dir),
                          "outputs", "inpaint-anything",
                          datetime.now().strftime("%Y-%m-%d"))

sam_dict = dict(sam_masks=None, mask_image=None, cnet=None, orig_image=None, pad_mask=None)

def update_ia_outputs_dir():
    """Update inpaint-anything outputs directory.
    
    Returns:
        None
    """
    global ia_outputs_dir
    config_save_folder = shared.opts.data.get("inpaint_anything_save_folder", "inpaint-anything")
    if config_save_folder in ["inpaint-anything", "img2img-images"]:
        ia_outputs_dir = os.path.join(os.path.dirname(extensions_dir),
                                      "outputs", config_save_folder,
                                      datetime.now().strftime("%Y-%m-%d"))

def save_mask_image(mask_image, save_mask_chk=False):
    """Save mask image.
    
    Args:
        mask_image (np.ndarray): mask image
        save_mask_chk (bool, optional): If True, save mask image. Defaults to False.
    
    Returns:
        None
    """
    global ia_outputs_dir
    if save_mask_chk:
        if not os.path.isdir(ia_outputs_dir):
            os.makedirs(ia_outputs_dir, exist_ok=True)
        save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "created_mask" + ".png"
        save_name = os.path.join(ia_outputs_dir, save_name)
        Image.fromarray(mask_image).save(save_name)

def pre_unload_model_weights():
    unload_model_weights()
    clear_cache()

def post_reload_model_weights():
    if shared.sd_model is None:
        reload_model_weights()

def clear_cache():
    gc.collect()
    torch_gc()

def sleep_clear_cache_and_reload_model():
    time.sleep(0.1)
    clear_cache()
    post_reload_model_weights()

def input_image_upload(input_image):
    clear_cache()
    global sam_dict
    sam_dict["orig_image"] = input_image
    sam_dict["pad_mask"] = None

def run_padding(input_image, pad_scale_width, pad_scale_height, pad_lr_barance, pad_tb_barance, padding_mode="edge"):
    clear_cache()
    global sam_dict
    if input_image is None or sam_dict["orig_image"] is None:
        sam_dict["orig_image"] = None
        sam_dict["pad_mask"] = None
        return None, "Input image not found"

    orig_image = sam_dict["orig_image"]

    height, width = orig_image.shape[:2]
    pad_width, pad_height = (int(width * pad_scale_width), int(height * pad_scale_height))
    ia_logging.info(f"resize by padding: ({height}, {width}) -> ({pad_height}, {pad_width})")

    pad_size_w, pad_size_h = (pad_width - width, pad_height - height)
    pad_size_l = int(pad_size_w * pad_lr_barance)
    pad_size_r = pad_size_w - pad_size_l
    pad_size_t = int(pad_size_h * pad_tb_barance)
    pad_size_b = pad_size_h - pad_size_t
    
    pad_width=[(pad_size_t, pad_size_b), (pad_size_l, pad_size_r), (0, 0)]
    if padding_mode == "constant":
        fill_value = shared.opts.data.get("inpaint_anything_padding_fill", 127)
        pad_image = np.pad(orig_image, pad_width=pad_width, mode=padding_mode, constant_values=fill_value)
    else:
        pad_image = np.pad(orig_image, pad_width=pad_width, mode=padding_mode)

    mask_pad_width = [(pad_size_t, pad_size_b), (pad_size_l, pad_size_r)]
    pad_mask = np.zeros((height, width), dtype=np.uint8)
    pad_mask = np.pad(pad_mask, pad_width=mask_pad_width, mode="constant", constant_values=255)
    sam_dict["pad_mask"] = dict(segmentation=pad_mask.astype(np.bool))

    return pad_image, "Padding done"

def run_sam(input_image, sam_model_id, sam_image):
    clear_cache()
    global sam_dict
    if sam_dict["sam_masks"] is not None:
        sam_dict["sam_masks"] = None
        clear_cache()
    
    sam_checkpoint = os.path.join(extensions_dir, "sd-webui-inpaint-anything", "models", sam_model_id)
    if not os.path.isfile(sam_checkpoint):
        return None, f"{sam_model_id} not found, please download"
    
    if input_image is None:
        return None, "Input image not found"
    
    pre_unload_model_weights()
    ia_logging.info(f"input_image: {input_image.shape} {input_image.dtype}")
    
    cm_pascal = create_pascal_label_colormap()
    seg_colormap = cm_pascal
    seg_colormap = [c for c in seg_colormap if max(c) >= 64]
    # print(len(seg_colormap))
    
    sam_mask_generator = get_sam_mask_generator(sam_checkpoint)
    ia_logging.info(f"{sam_mask_generator.__class__.__name__} {sam_model_id}")
    sam_masks = sam_mask_generator.generate(input_image)

    canvas_image = np.zeros_like(input_image)

    ia_logging.info("sam_masks: {}".format(len(sam_masks)))
    sam_masks = sorted(sam_masks, key=lambda x: np.sum(x.get("segmentation").astype(int)))
    if sam_dict["pad_mask"] is not None:
        if len(sam_masks) > 0 and sam_masks[0]["segmentation"].shape == sam_dict["pad_mask"]["segmentation"].shape:
            sam_masks.insert(0, sam_dict["pad_mask"])
            ia_logging.info("insert pad_mask to sam_masks")
    sam_masks = sam_masks[:len(seg_colormap)]
    for idx, seg_dict in enumerate(sam_masks):
        seg_mask = np.expand_dims(seg_dict["segmentation"].astype(int), axis=-1)
        canvas_mask = np.logical_not(np.sum(canvas_image, axis=-1, keepdims=True).astype(bool)).astype(int)
        seg_color = seg_colormap[idx] * seg_mask * canvas_mask
        canvas_image = canvas_image + seg_color
    seg_image = canvas_image.astype(np.uint8)
    
    sam_dict["sam_masks"] = sam_masks

    del sam_mask_generator
    if sam_image is None:
        return seg_image, "Segment Anything complete"
    else:
        if sam_image["image"].shape == seg_image.shape and np.all(sam_image["image"] == seg_image):
            return gr.update(), "Segment Anything complete"
        else:
            return gr.update(value=seg_image), "Segment Anything complete"

def select_mask(input_image, sam_image, invert_chk, sel_mask):
    clear_cache()
    global sam_dict
    if sam_dict["sam_masks"] is None or sam_image is None:
        return None
    sam_masks = sam_dict["sam_masks"]
    
    image = sam_image["image"]
    mask = sam_image["mask"][:,:,0:3]
    
    canvas_image = np.zeros_like(image)
    mask_region = np.zeros_like(image)
    for idx, seg_dict in enumerate(sam_masks):
        seg_mask = np.expand_dims(seg_dict["segmentation"].astype(int), axis=-1)
        canvas_mask = np.logical_not(np.sum(canvas_image, axis=-1, keepdims=True).astype(bool)).astype(int)
        if (seg_mask * canvas_mask * mask).astype(bool).any():
            mask_region = mask_region + (seg_mask * canvas_mask * 255)
        # seg_color = seg_colormap[idx] * seg_mask * canvas_mask
        seg_color = [127, 127, 127] * seg_mask * canvas_mask
        canvas_image = canvas_image + seg_color
    
    canvas_mask = np.logical_not(np.sum(canvas_image, axis=-1, keepdims=True).astype(bool)).astype(int)
    if (canvas_mask * mask).astype(bool).any():
        mask_region = mask_region + (canvas_mask * 255)
    
    seg_image = mask_region.astype(np.uint8)

    if invert_chk:
        seg_image = np.logical_not(seg_image.astype(bool)).astype(np.uint8) * 255

    sam_dict["mask_image"] = seg_image

    if input_image is not None and input_image.shape == seg_image.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, seg_image, 0.5, 0)
    else:
        ret_image = seg_image

    clear_cache()
    if sel_mask is None:
        return ret_image
    else:
        if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
            return gr.update()
        else:
            return gr.update(value=ret_image)

def expand_mask(input_image, sel_mask, expand_iteration=1):
    clear_cache()
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None
    
    new_sel_mask = sam_dict["mask_image"]
    
    expand_iteration = int(np.clip(expand_iteration, 1, 5))
    
    new_sel_mask = cv2.dilate(new_sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=expand_iteration)
    
    sam_dict["mask_image"] = new_sel_mask

    if input_image is not None and input_image.shape == new_sel_mask.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
    else:
        ret_image = new_sel_mask

    clear_cache()
    if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
        return gr.update()
    else:
        return gr.update(value=ret_image)

def apply_mask(input_image, sel_mask):
    clear_cache()
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None
    
    sel_mask_image = sam_dict["mask_image"]
    sel_mask_mask = np.logical_not(sel_mask["mask"][:,:,0:3].astype(bool)).astype(np.uint8)
    new_sel_mask = sel_mask_image * sel_mask_mask
    
    sam_dict["mask_image"] = new_sel_mask

    if input_image is not None and input_image.shape == new_sel_mask.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
    else:
        ret_image = new_sel_mask

    clear_cache()
    if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
        return gr.update()
    else:
        return gr.update(value=ret_image)

def auto_resize_to_pil(input_image, mask_image):
    init_image = Image.fromarray(input_image).convert("RGB")
    mask_image = Image.fromarray(mask_image).convert("RGB")
    assert init_image.size == mask_image.size, "The size of image and mask do not match"
    width, height = init_image.size

    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    if new_width < width or new_height < height:
        if (new_width / width) < (new_height / height):
            scale = new_height / height
        else:
            scale = new_width / width
        resize_height = int(height*scale+0.5)
        resize_width = int(width*scale+0.5)
        ia_logging.info(f"resize: ({height}, {width}) -> ({resize_height}, {resize_width})")
        init_image = transforms.functional.resize(init_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
        mask_image = transforms.functional.resize(mask_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
        ia_logging.info(f"center_crop: ({resize_height}, {resize_width}) -> ({new_height}, {new_width})")
        init_image = transforms.functional.center_crop(init_image, (new_height, new_width))
        mask_image = transforms.functional.center_crop(mask_image, (new_height, new_width))
        assert init_image.size == mask_image.size, "The size of image and mask do not match"
    
    return init_image, mask_image

def run_inpaint(input_image, sel_mask, prompt, n_prompt, ddim_steps, cfg_scale, seed, model_id, save_mask_chk, composite_chk, sampler_name="DDIM"):
    clear_cache()
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        return None

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.warning("The size of image and mask do not match")
        return None

    global ia_outputs_dir
    update_ia_outputs_dir()
    save_mask_image(mask_image, save_mask_chk)

    pre_unload_model_weights()

    ia_logging.info(f"Loading model {model_id}")
    config_offline_inpainting = shared.opts.data.get("inpaint_anything_offline_inpainting", False)
    if config_offline_inpainting:
        ia_logging.info("Enable offline network Inpainting: {}".format(str(config_offline_inpainting)))
    local_files_only = False
    local_file_status = download_model_from_hf(model_id, local_files_only=True)
    if local_file_status != _DOWNLOAD_COMPLETE:
        if config_offline_inpainting:
            ia_logging.warning(local_file_status)
            return None
    else:
        local_files_only = True
        ia_logging.info("local_files_only: {}".format(str(local_files_only)))
    
    if platform.system() == "Darwin":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16
    
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, local_files_only=local_files_only)
    except Exception as e:
        ia_logging.error(str(e))
        if not config_offline_inpainting:
            try:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, resume_download=True)
            except Exception as e:
                ia_logging.error(str(e))
                try:
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, force_download=True)
                except Exception as e:
                    ia_logging.error(str(e))
                    return None
        else:
            return None
    pipe.safety_checker = None

    ia_logging.info(f"Using sampler {sampler_name}")
    if sampler_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "DPM2 Karras":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "DPM2 a Karras":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        ia_logging.info("Sampler fallback to DDIM")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    if seed < 0:
        seed = random.randint(0, 2147483647)
    
    if platform.system() == "Darwin":
        pipe = pipe.to("mps")
        pipe.enable_attention_slicing()
        generator = torch.Generator("cpu").manual_seed(seed)
    else:
        # pipe.enable_model_cpu_offload()
        pipe = pipe.to(device)
        if shared.xformers_available:
            pipe.enable_xformers_memory_efficient_attention()
        else:
            pipe.enable_attention_slicing()
        generator = torch.Generator(device).manual_seed(seed)
    
    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size
    
    pipe_args_dict = {
        "prompt": prompt,
        "image": init_image,
        "width": width,
        "height": height,
        "mask_image": mask_image,
        "num_inference_steps": ddim_steps,
        "guidance_scale": cfg_scale,
        "negative_prompt": n_prompt,
        "generator": generator,
        }
    
    output_image = pipe(**pipe_args_dict).images[0]
    
    if composite_chk:
        mask_image = Image.fromarray(cv2.dilate(np.array(mask_image), np.ones((3, 3), dtype=np.uint8), iterations=4))
        output_image = Image.composite(output_image, init_image, mask_image.convert("L").filter(ImageFilter.GaussianBlur(3)))

    generation_params = {
        "Steps": ddim_steps,
        "Sampler": pipe.scheduler.__class__.__name__,
        "CFG scale": cfg_scale,
        "Seed": seed,
        "Size": f"{width}x{height}",
        "Model": model_id,
        }

    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])
    prompt_text = prompt if prompt else ""
    negative_prompt_text = "Negative prompt: " + n_prompt if n_prompt else ""
    infotext = f"{prompt_text}\n{negative_prompt_text}\n{generation_params_text}".strip()
    
    metadata = PngInfo()
    metadata.add_text("parameters", infotext)
    
    if not os.path.isdir(ia_outputs_dir):
        os.makedirs(ia_outputs_dir, exist_ok=True)
    save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + os.path.basename(model_id) + "_" + str(seed) + ".png"
    save_name = os.path.join(ia_outputs_dir, save_name)
    output_image.save(save_name, pnginfo=metadata)
    
    del pipe
    return output_image

def run_cleaner(input_image, sel_mask, cleaner_model_id, cleaner_save_mask_chk):
    clear_cache()
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        return None
    
    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.warning("The size of image and mask do not match")
        return None

    global ia_outputs_dir
    update_ia_outputs_dir()
    save_mask_image(mask_image, cleaner_save_mask_chk)

    pre_unload_model_weights()

    ia_logging.info(f"Loading model {cleaner_model_id}")
    if platform.system() == "Darwin":
        model = ModelManager(name=cleaner_model_id, device="cpu")
    else:
        model = ModelManager(name=cleaner_model_id, device=device)
    
    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size
    
    init_image = np.array(init_image)
    mask_image = np.array(mask_image.convert("L"))
    
    config = Config(
        ldm_steps=20,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_resize_limit=512,
        prompt="",
        sd_steps=20,
        sd_sampler=SDSampler.ddim
    )
    
    output_image = model(image=init_image, mask=mask_image, config=config)
    # print(output_image.shape, output_image.dtype, np.min(output_image), np.max(output_image))
    output_image = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output_image)

    if not os.path.isdir(ia_outputs_dir):
        os.makedirs(ia_outputs_dir, exist_ok=True)
    save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + os.path.basename(cleaner_model_id) + ".png"
    save_name = os.path.join(ia_outputs_dir, save_name)
    output_image.save(save_name)
    
    del model
    return output_image

def run_get_alpha_image(input_image, sel_mask):
    clear_cache()
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        return None
    
    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.warning("The size of image and mask do not match")
        return None

    alpha_image = Image.fromarray(input_image).convert("RGBA")
    mask_image = Image.fromarray(mask_image).convert("L")
    
    alpha_image.putalpha(mask_image)
    
    global ia_outputs_dir
    update_ia_outputs_dir()
    if not os.path.isdir(ia_outputs_dir):
        os.makedirs(ia_outputs_dir, exist_ok=True)
    save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "rgba_image" + ".png"
    save_name = os.path.join(ia_outputs_dir, save_name)
    alpha_image.save(save_name)
    
    def make_checkerboard(n_rows, n_columns, square_size):
        n_rows_, n_columns_ = int(n_rows/square_size + 1), int(n_columns/square_size + 1)
        rows_grid, columns_grid = np.meshgrid(range(n_rows_), range(n_columns_), indexing='ij')
        high_res_checkerboard = (np.mod(rows_grid, 2) + np.mod(columns_grid, 2)) == 1
        square = np.ones((square_size,square_size))
        checkerboard = np.kron(high_res_checkerboard, square)[:n_rows,:n_columns]

        return checkerboard
    
    checkerboard = make_checkerboard(alpha_image.size[1], alpha_image.size[0], 16)
    checkerboard = np.clip((checkerboard * 255), 128, 192).astype(np.uint8)
    checkerboard = Image.fromarray(checkerboard).convert("RGBA")
    checkerboard.putalpha(ImageOps.invert(mask_image))
    
    output_image = Image.alpha_composite(alpha_image, checkerboard)
    
    clear_cache()
    return output_image

def run_get_mask(sel_mask):
    clear_cache()
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None
    
    mask_image = sam_dict["mask_image"]

    global ia_outputs_dir
    update_ia_outputs_dir()
    if not os.path.isdir(ia_outputs_dir):
        os.makedirs(ia_outputs_dir, exist_ok=True)
    save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "created_mask" + ".png"
    save_name = os.path.join(ia_outputs_dir, save_name)
    Image.fromarray(mask_image).save(save_name)
    
    clear_cache()
    return mask_image

def run_cn_inpaint(input_image, sel_mask,
                   cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed, cn_module_id, cn_model_id, cn_save_mask_chk,
                   cn_low_vram_chk, cn_weight, cn_mode, cn_ref_module_id=None, cn_ref_image=None, cn_ref_weight=1.0, cn_ref_mode="Balanced"):
    clear_cache()
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        return None

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.warning("The size of image and mask do not match")
        return None

    if shared.sd_model is None:
        reload_model_weights()

    if (shared.sd_model.parameterization == "v" and "sd15" in cn_model_id):
        ia_logging.warning("The SD model is not compatible with the ControlNet model")
        return None

    cnet = sam_dict.get("cnet", None)
    if cnet is None:
        ia_logging.warning("The ControlNet extension is not loaded")
        return None

    global ia_outputs_dir
    update_ia_outputs_dir()
    save_mask_image(mask_image, cn_save_mask_chk)

    # print(cn_model_id)
    if cn_seed < 0:
        cn_seed = random.randint(0, 2147483647)
    
    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    p = get_sd_img2img_processing(init_image, None, cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed)

    backup_alwayson_scripts(p.scripts)
    disable_alwayson_scripts(p.scripts)

    cn_units = [cnet.ControlNetUnit(
        enabled=True,
        module=cn_module_id,
        model=cn_model_id,
        weight=cn_weight,
        image={"image": np.array(init_image), "mask": np.array(mask_image)},
        resize_mode=cnet.ResizeMode.RESIZE,
        low_vram=cn_low_vram_chk,
        processor_res=min(width, height),
        guidance_start=0.0,
        guidance_end=1.0,
        pixel_perfect=True,
        control_mode=cn_mode,
    )]
    
    if cn_ref_module_id is not None and cn_ref_image is not None:
        cn_ref_image = resize_image(1, Image.fromarray(cn_ref_image), width=width, height=height)
        
        cn_units.append(cnet.ControlNetUnit(
            enabled=True,
            module=cn_ref_module_id,
            model=None,
            weight=cn_ref_weight,
            image=np.array(cn_ref_image),
            resize_mode=cnet.ResizeMode.RESIZE,
            low_vram=cn_low_vram_chk,
            processor_res=min(width, height),
            guidance_start=0.0,
            guidance_end=1.0,
            pixel_perfect=True,
            control_mode=cn_ref_mode,
        ))
    
    p.script_args = np.zeros(get_controlnet_args_to(p.scripts))
    cnet.update_cn_script_in_processing(p, cn_units)

    processed = process_images(p)

    clear_controlnet_cache(p.scripts)
    restore_alwayson_scripts(p.scripts)

    no_hash_cn_model_id = re.sub("\s\[[0-9a-f]{8,10}\]", "", cn_model_id).strip()

    if processed is not None:
        if len(processed.images) > 0:
            output_image = processed.images[0]

            infotext = create_infotext(p, all_prompts=[cn_prompt], all_seeds=[cn_seed], all_subseeds=[-1])

            metadata = PngInfo()
            metadata.add_text("parameters", infotext)

            if not os.path.isdir(ia_outputs_dir):
                os.makedirs(ia_outputs_dir, exist_ok=True)
            save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + os.path.basename(no_hash_cn_model_id) + "_" + str(cn_seed) + ".png"
            save_name = os.path.join(ia_outputs_dir, save_name)
            output_image.save(save_name, pnginfo=metadata)
        else:
            output_image = None
    else:
        output_image = None

    return output_image

# class Script(scripts.Script):
#   def __init__(self) -> None:
#     super().__init__()

#   def title(self):
#     return "Inpaint Anything"

#   def show(self, is_img2img):
#     return scripts.AlwaysVisible

#   def ui(self, is_img2img):
#     return ()

def on_ui_tabs():
    global sam_dict
    
    sampler_names = get_sampler_names()
    sam_model_ids = get_sam_model_ids()
    sam_model_index =  sam_model_ids.index("sam_vit_l_0b3195.pth") if "sam_vit_l_0b3195.pth" in sam_model_ids else 1
    model_ids = get_model_ids()
    cleaner_model_ids = get_cleaner_model_ids()
    padding_mode_names = get_padding_mode_names()
    sam_dict["cnet"] = find_controlnet()

    cn_enabled = False
    if sam_dict["cnet"] is not None:
        cn_module_ids = [cn for cn in sam_dict["cnet"].get_modules() if "inpaint" in cn]
        cn_module_index = cn_module_ids.index("inpaint_only") if "inpaint_only" in cn_module_ids else 0

        cn_model_ids = [cn for cn in sam_dict["cnet"].get_models() if "inpaint" in cn]
        cn_modes = [mode.value for mode in sam_dict["cnet"].ControlMode]

        if len(cn_module_ids) > 0 and len(cn_model_ids) > 0:
            cn_enabled = True

    if samplers_for_img2img is not None and len(samplers_for_img2img) > 0:
        cn_sampler_ids = [sampler.name for sampler in samplers_for_img2img]
    else:
        cn_sampler_ids = ["DDIM"]
    cn_sampler_index = cn_sampler_ids.index("DDIM") if "DDIM" in cn_sampler_ids else -1

    cn_ref_only = False
    if cn_enabled and sam_dict["cnet"].get_max_models_num() > 1:
        cn_ref_module_ids = [cn for cn in sam_dict["cnet"].get_modules() if "reference" in cn]
        if len(cn_ref_module_ids) > 0:
            cn_ref_only = True

    with gr.Blocks(analytics_enabled=False) as inpaint_anything_interface:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        sam_model_id = gr.Dropdown(label="Segment Anything Model ID", elem_id="sam_model_id", choices=sam_model_ids,
                                                   value=sam_model_ids[sam_model_index], show_label=True)
                    with gr.Column():
                        with gr.Row():
                            load_model_btn = gr.Button("Download model", elem_id="load_model_btn")
                        with gr.Row():
                            status_text = gr.Textbox(label="", max_lines=1, show_label=False, interactive=False)
                with gr.Row():
                    input_image = gr.Image(label="Input image", elem_id="input_image", source="upload", type="numpy", interactive=True)
                
                with gr.Row():
                    with gr.Accordion("Padding options", elem_id="padding_options", open=False):
                        with gr.Row():
                            with gr.Column():
                                pad_scale_width = gr.Slider(label="Scale Width", elem_id="pad_scale_width", minimum=1.0, maximum=1.5, value=1.0, step=0.01)
                            with gr.Column():
                                pad_lr_barance = gr.Slider(label="Left/Right Balance", elem_id="pad_lr_barance", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                        with gr.Row():
                            with gr.Column():
                                pad_scale_height = gr.Slider(label="Scale Height", elem_id="pad_scale_height", minimum=1.0, maximum=1.5, value=1.0, step=0.01)
                            with gr.Column():
                                pad_tb_barance = gr.Slider(label="Top/Bottom Balance", elem_id="pad_tb_barance", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                        with gr.Row():
                            with gr.Column():
                                padding_mode = gr.Dropdown(label="Padding Mode", elem_id="padding_mode", choices=padding_mode_names, value="edge")
                            with gr.Column():
                                padding_btn = gr.Button("Run Padding", elem_id="padding_btn")
                
                with gr.Row():
                    sam_btn = gr.Button("Run Segment Anything", elem_id="sam_btn")
                
                with gr.Tab("Inpainting", elem_id="inpainting_tab"):
                    prompt = gr.Textbox(label="Inpainting Prompt", elem_id="sd_prompt")
                    n_prompt = gr.Textbox(label="Negative Prompt", elem_id="sd_n_prompt")
                    with gr.Accordion("Advanced options", elem_id="inp_advanced_options", open=False):
                        with gr.Row():
                            with gr.Column():
                                sampler_name = gr.Dropdown(label="Sampler", elem_id="sampler_name", choices=sampler_names,
                                                           value=sampler_names[0], show_label=True)
                            with gr.Column():
                                ddim_steps = gr.Slider(label="Sampling Steps", elem_id="ddim_steps", minimum=1, maximum=100, value=20, step=1)
                        cfg_scale = gr.Slider(label="Guidance Scale", elem_id="cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                        seed = gr.Slider(
                            label="Seed",
                            elem_id="sd_seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1,
                        )
                    with gr.Row():
                        with gr.Column():
                            model_id = gr.Dropdown(label="Inpainting Model ID", elem_id="model_id", choices=model_ids, value=model_ids[0], show_label=True)
                        with gr.Column():
                            with gr.Row():
                                inpaint_btn = gr.Button("Run Inpainting", elem_id="inpaint_btn")
                            with gr.Row():
                                composite_chk = gr.Checkbox(label="Mask area Only", elem_id="composite_chk", value=True, show_label=True, interactive=True)
                                save_mask_chk = gr.Checkbox(label="Save mask", elem_id="save_mask_chk", show_label=True, interactive=True)

                    with gr.Row():
                        out_image = gr.Image(label="Inpainted image", elem_id="out_image", type="pil", interactive=False).style(height=480)
                
                with gr.Tab("Cleaner", elem_id="cleaner_tab"):
                    with gr.Row():
                        with gr.Column():
                            cleaner_model_id = gr.Dropdown(label="Cleaner Model ID", elem_id="cleaner_model_id", choices=cleaner_model_ids, value=cleaner_model_ids[0], show_label=True)
                        with gr.Column():
                            with gr.Row():
                                cleaner_btn = gr.Button("Run Cleaner", elem_id="cleaner_btn")
                            with gr.Row():
                                cleaner_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="cleaner_save_mask_chk", show_label=True, interactive=True)
                    
                    with gr.Row():
                        cleaner_out_image = gr.Image(label="Cleaned image", elem_id="cleaner_out_image", type="pil", interactive=False).style(height=480)

                with gr.Tab("ControlNet Inpaint", elem_id="cn_inpaint_tab"):
                    if cn_enabled:
                        cn_prompt = gr.Textbox(label="Inpainting Prompt", elem_id="cn_sd_prompt")
                        cn_n_prompt = gr.Textbox(label="Negative Prompt", elem_id="cn_sd_n_prompt")
                        with gr.Accordion("Advanced options", elem_id="cn_advanced_options", open=False):
                            with gr.Row():
                                with gr.Column():
                                    cn_sampler_id = gr.Dropdown(label="Sampling method", elem_id="cn_sampler_id", choices=cn_sampler_ids, value=cn_sampler_ids[cn_sampler_index], show_label=True)
                                with gr.Column():
                                    cn_ddim_steps = gr.Slider(label="Sampling steps", elem_id="cn_ddim_steps", minimum=1, maximum=150, value=30, step=1)
                            cn_cfg_scale = gr.Slider(label="Guidance scale", elem_id="cn_cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                            cn_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.6, elem_id="cn_strength")
                            cn_seed = gr.Slider(
                                label="Seed",
                                elem_id="cn_sd_seed",
                                minimum=-1,
                                maximum=2147483647,
                                step=1,
                                value=-1,
                            )
                        with gr.Accordion("ControlNet options", elem_id="cn_cn_options", open=False):
                            with gr.Row():
                                with gr.Column():
                                    cn_low_vram_chk = gr.Checkbox(label="Low VRAM", elem_id="cn_low_vram_chk", show_label=True, interactive=True)
                                    cn_weight = gr.Slider(label="Control Weight", elem_id="cn_weight", minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                                with gr.Column():
                                    cn_mode = gr.Dropdown(label="Control Mode", elem_id="cn_mode", choices=cn_modes, value=cn_modes[-1], show_label=True)
                                
                            if cn_ref_only:
                                with gr.Row():
                                    gr.Markdown("Reference-Only Control (enabled only when a reference image below is present)")
                                with gr.Row():
                                    with gr.Column():
                                        cn_ref_image = gr.Image(label="Reference Image", elem_id="cn_ref_image", source="upload", type="numpy", interactive=True)
                                    with gr.Column():
                                        cn_ref_module_id = gr.Dropdown(label="Reference Type", elem_id="cn_ref_module_id", choices=cn_ref_module_ids, value=cn_ref_module_ids[-1], show_label=True)
                                        cn_ref_weight = gr.Slider(label="Reference Control Weight", elem_id="cn_ref_weight", minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                                        cn_ref_mode = gr.Dropdown(label="Reference Control Mode", elem_id="cn_ref_mode", choices=cn_modes, value=cn_modes[0], show_label=True)
                            else:
                                with gr.Row():
                                    gr.Markdown("The Multi ControlNet setting is currently set to 1.<br>" + \
                                        "If you wish to use the Reference-Only Control, please adjust the Multi ControlNet setting to 2 or more and restart the Web UI.")

                        with gr.Row():
                            with gr.Column():
                                cn_module_id = gr.Dropdown(label="ControlNet Preprocessor", elem_id="cn_module_id", choices=cn_module_ids, value=cn_module_ids[cn_module_index], show_label=True)
                                cn_model_id = gr.Dropdown(label="ControlNet Model ID", elem_id="cn_model_id", choices=cn_model_ids, value=cn_model_ids[0], show_label=True)
                            with gr.Column():
                                with gr.Row():
                                    cn_inpaint_btn = gr.Button("Run ControlNet Inpaint", elem_id="cn_inpaint_btn")
                                with gr.Row():
                                    cn_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="cn_save_mask_chk", show_label=True, interactive=True)
                        
                        with gr.Row():
                            cn_out_image = gr.Image(label="Inpainted image", elem_id="cn_out_image", type="pil", interactive=False).style(height=480)
                        
                    else:
                        if sam_dict["cnet"] is None:
                            gr.Markdown("ControlNet extension is not available.<br>" + \
                                        "Requires the [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension.")
                        elif len(cn_module_ids) > 0:
                            cn_models_directory = os.path.join("extensions", "sd-webui-controlnet", "models")
                            gr.Markdown("ControlNet inpaint model is not available.<br>" + \
                                        f"Requires the [ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1) inpaint model in the {cn_models_directory} directory.")
                        else:
                            gr.Markdown("ControlNet inpaint preprocessor is not available.<br>" + \
                                        "The local version of [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension may be old.")

                with gr.Tab("Mask only", elem_id="mask_only_tab"):
                    with gr.Row():
                        with gr.Column():
                            get_alpha_image_btn = gr.Button("Get mask as alpha of image", elem_id="get_alpha_image_btn")
                        with gr.Column():
                            get_mask_btn = gr.Button("Get mask", elem_id="get_mask_btn")
                    
                    with gr.Row():
                        with gr.Column():
                            alpha_out_image = gr.Image(label="Alpha channel image", elem_id="alpha_out_image", type="pil", interactive=False)
                        with gr.Column():
                            mask_out_image = gr.Image(label="Mask image", elem_id="mask_out_image", type="numpy", interactive=False)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("")
                        with gr.Column():
                            mask_send_to_inpaint_btn = gr.Button("Send to img2img inpaint", elem_id="mask_send_to_inpaint_btn")
            
            with gr.Column():
                sam_image = gr.Image(label="Segment Anything image", elem_id="sam_image", type="numpy", tool="sketch", brush_radius=8,
                                     interactive=True).style(height=480)
                with gr.Row():
                    with gr.Column():
                        select_btn = gr.Button("Create mask", elem_id="select_btn")
                    with gr.Column():
                        invert_chk = gr.Checkbox(label="Invert mask", elem_id="invert_chk", show_label=True, interactive=True)

                sel_mask = gr.Image(label="Selected mask image", elem_id="sel_mask", type="numpy", tool="sketch", brush_radius=12,
                                    interactive=True).style(height=480)

                with gr.Row():
                    with gr.Column():
                        expand_mask_btn = gr.Button("Expand mask region", elem_id="expand_mask_btn")
                    with gr.Column():
                        apply_mask_btn = gr.Button("Trim mask by sketch", elem_id="apply_mask_btn")
            
            load_model_btn.click(download_model, inputs=[sam_model_id], outputs=[status_text])
            input_image.upload(input_image_upload, inputs=[input_image], outputs=None)
            padding_btn.click(run_padding, inputs=[input_image, pad_scale_width, pad_scale_height, pad_lr_barance, pad_tb_barance, padding_mode], outputs=[input_image, status_text])
            sam_btn.click(run_sam, inputs=[input_image, sam_model_id, sam_image], outputs=[sam_image, status_text]).then(
                fn=sleep_clear_cache_and_reload_model, inputs=None, outputs=None, _js="inpaintAnything_clearSamMask")
            select_btn.click(select_mask, inputs=[input_image, sam_image, invert_chk, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            expand_mask_btn.click(expand_mask, inputs=[input_image, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            apply_mask_btn.click(apply_mask, inputs=[input_image, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            inpaint_btn.click(
                run_inpaint,
                inputs=[input_image, sel_mask, prompt, n_prompt, ddim_steps, cfg_scale, seed, model_id, save_mask_chk, composite_chk, sampler_name],
                outputs=[out_image]).then(
                fn=sleep_clear_cache_and_reload_model, inputs=None, outputs=None)
            cleaner_btn.click(
                run_cleaner,
                inputs=[input_image, sel_mask, cleaner_model_id, cleaner_save_mask_chk],
                outputs=[cleaner_out_image]).then(
                fn=sleep_clear_cache_and_reload_model, inputs=None, outputs=None)
            get_alpha_image_btn.click(
                run_get_alpha_image,
                inputs=[input_image, sel_mask],
                outputs=[alpha_out_image])
            get_mask_btn.click(
                run_get_mask,
                inputs=[sel_mask],
                outputs=[mask_out_image])
            mask_send_to_inpaint_btn.click(
                fn=None,
                _js="inpaintAnything_sendToInpaint",
                inputs=None,
                outputs=None)
            if cn_enabled and not cn_ref_only:
                cn_inpaint_btn.click(
                    run_cn_inpaint,
                    inputs=[input_image, sel_mask,
                            cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed, cn_module_id, cn_model_id, cn_save_mask_chk,
                            cn_low_vram_chk, cn_weight, cn_mode],
                    outputs=[cn_out_image]).then(
                    fn=sleep_clear_cache_and_reload_model, inputs=None, outputs=None)
            elif cn_enabled and cn_ref_only:
                cn_inpaint_btn.click(
                    run_cn_inpaint,
                    inputs=[input_image, sel_mask,
                            cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed, cn_module_id, cn_model_id, cn_save_mask_chk,
                            cn_low_vram_chk, cn_weight, cn_mode, cn_ref_module_id, cn_ref_image, cn_ref_weight, cn_ref_mode],
                    outputs=[cn_out_image]).then(
                    fn=sleep_clear_cache_and_reload_model, inputs=None, outputs=None)

    return [(inpaint_anything_interface, "Inpaint Anything", "inpaint_anything")]

def on_ui_settings():
    section = ("inpaint_anything", "Inpaint Anything")
    shared.opts.add_option("inpaint_anything_save_folder", shared.OptionInfo(
        "inpaint-anything", "Folder name where output images will be saved", gr.Radio, {"choices": ["inpaint-anything", "img2img-images"]}, section=section))
    shared.opts.add_option("inpaint_anything_offline_inpainting", shared.OptionInfo(
        False, "Enable offline network Inpainting", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("inpaint_anything_padding_fill", shared.OptionInfo(
        127, "Fill value used when Padding is set to constant", gr.Slider, {"minimum":0, "maximum":255, "step":1}, section=section))

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
