import gc
import math
import os
import platform

if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
import re
import traceback

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import (DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler,
                       StableDiffusionInpaintPipeline)
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler
from modules import devices, script_callbacks, shared
from modules.processing import create_infotext, process_images
from modules.sd_models import get_closet_checkpoint_match
from modules.sd_samplers import samplers_for_img2img
from PIL import Image, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo
from torch.hub import download_url_to_file
from torchvision import transforms

import inpalib
from ia_check_versions import ia_check_versions
from ia_config import (IAConfig, get_ia_config_index, get_webui_setting, set_ia_config,
                       setup_ia_config_ini)
from ia_file_manager import IAFileManager, download_model_from_hf, ia_file_manager
from ia_logging import draw_text_image, ia_logging
from ia_threading import (async_post_reload_model_weights, await_backup_reload_ckpt_info,
                          await_pre_reload_model_weights, clear_cache_decorator,
                          clear_cache_yield_decorator, offload_reload_decorator,
                          offload_reload_yield_decorator)
from ia_ui_items import (get_cleaner_model_ids, get_inp_model_ids, get_inp_webui_model_ids,
                         get_padding_mode_names, get_sam_model_ids, get_sampler_names)
from ia_webui_controlnet import (backup_alwayson_scripts, clear_controlnet_cache,
                                 disable_all_alwayson_scripts, disable_alwayson_scripts_wo_cn,
                                 find_controlnet, get_controlnet_args_to, get_max_args_to,
                                 get_sd_img2img_processing, restore_alwayson_scripts)


@clear_cache_decorator
def download_model(sam_model_id):
    """Download SAM model.

    Args:
        sam_model_id (str): SAM model id

    Returns:
        str: download status
    """
    if "_hq_" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/sam-hq/resolve/main/" + sam_model_id
    elif "FastSAM" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/FastSAM/resolve/main/" + sam_model_id
    elif "mobile_sam" in sam_model_id:
        url_sam = "https://huggingface.co/Uminosachi/MobileSAM/resolve/main/" + sam_model_id
    else:
        # url_sam_vit_h_4b8939 = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        url_sam = "https://dl.fbaipublicfiles.com/segment_anything/" + sam_model_id

    sam_checkpoint = os.path.join(ia_file_manager.models_dir, sam_model_id)
    if not os.path.isfile(sam_checkpoint):
        try:
            download_url_to_file(url_sam, sam_checkpoint)
        except Exception as e:
            ia_logging.error(str(e))
            return str(e)

        return IAFileManager.DOWNLOAD_COMPLETE
    else:
        return "Model already exists"


sam_dict = dict(sam_masks=None, mask_image=None, cnet=None, orig_image=None, pad_mask=None)


def save_mask_image(mask_image, save_mask_chk=False):
    """Save mask image.

    Args:
        mask_image (np.ndarray): mask image
        save_mask_chk (bool, optional): If True, save mask image. Defaults to False.

    Returns:
        None
    """
    if save_mask_chk:
        save_name = "_".join([ia_file_manager.savename_prefix, "created_mask"]) + ".png"
        save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
        Image.fromarray(mask_image).save(save_name)


@clear_cache_decorator
def input_image_upload(input_image, sam_image, sel_mask):
    global sam_dict
    sam_dict["orig_image"] = input_image
    sam_dict["pad_mask"] = None

    ret_sam_image = np.zeros_like(input_image, dtype=np.uint8) if sam_image is None else gr.update()
    ret_sel_mask = np.zeros_like(input_image, dtype=np.uint8) if sel_mask is None else gr.update()

    return ret_sam_image, ret_sel_mask, gr.update(interactive=True)


@clear_cache_decorator
def run_padding(input_image, pad_scale_width, pad_scale_height, pad_lr_barance, pad_tb_barance, padding_mode="edge"):
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

    pad_width = [(pad_size_t, pad_size_b), (pad_size_l, pad_size_r), (0, 0)]
    if padding_mode == "constant":
        fill_value = get_webui_setting("inpaint_anything_padding_fill", 127)
        pad_image = np.pad(orig_image, pad_width=pad_width, mode=padding_mode, constant_values=fill_value)
    else:
        pad_image = np.pad(orig_image, pad_width=pad_width, mode=padding_mode)

    mask_pad_width = [(pad_size_t, pad_size_b), (pad_size_l, pad_size_r)]
    pad_mask = np.zeros((height, width), dtype=np.uint8)
    pad_mask = np.pad(pad_mask, pad_width=mask_pad_width, mode="constant", constant_values=255)
    sam_dict["pad_mask"] = dict(segmentation=pad_mask.astype(bool))

    return pad_image, "Padding done"


@offload_reload_decorator
@clear_cache_decorator
def run_sam(input_image, sam_model_id, sam_image, anime_style_chk=False):
    global sam_dict
    if not inpalib.sam_file_exists(sam_model_id):
        ret_sam_image = None if sam_image is None else gr.update()
        return ret_sam_image, f"{sam_model_id} not found, please download"

    if input_image is None:
        ret_sam_image = None if sam_image is None else gr.update()
        return ret_sam_image, "Input image not found"

    set_ia_config(IAConfig.KEYS.SAM_MODEL_ID, sam_model_id, IAConfig.SECTIONS.USER)

    if sam_dict["sam_masks"] is not None:
        sam_dict["sam_masks"] = None
        gc.collect()

    ia_logging.info(f"input_image: {input_image.shape} {input_image.dtype}")

    try:
        sam_masks = inpalib.generate_sam_masks(input_image, sam_model_id, anime_style_chk)
        sam_masks = inpalib.sort_masks_by_area(sam_masks)
        sam_masks = inpalib.insert_mask_to_sam_masks(sam_masks, sam_dict["pad_mask"])

        seg_image = inpalib.create_seg_color_image(input_image, sam_masks)

        sam_dict["sam_masks"] = sam_masks

    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        ret_sam_image = None if sam_image is None else gr.update()
        return ret_sam_image, "Segment Anything failed"

    if sam_image is None:
        return seg_image, "Segment Anything complete"
    else:
        if sam_image["image"].shape == seg_image.shape and np.all(sam_image["image"] == seg_image):
            return gr.update(), "Segment Anything complete"
        else:
            return gr.update(value=seg_image), "Segment Anything complete"


@clear_cache_decorator
def select_mask(input_image, sam_image, invert_chk, ignore_black_chk, sel_mask):
    global sam_dict
    if sam_dict["sam_masks"] is None or sam_image is None:
        ret_sel_mask = None if sel_mask is None else gr.update()
        return ret_sel_mask
    sam_masks = sam_dict["sam_masks"]

    # image = sam_image["image"]
    mask = sam_image["mask"][:, :, 0:1]

    try:
        seg_image = inpalib.create_mask_image(mask, sam_masks, ignore_black_chk)
        if invert_chk:
            seg_image = inpalib.invert_mask(seg_image)

        sam_dict["mask_image"] = seg_image

    except Exception as e:
        print(traceback.format_exc())
        ia_logging.error(str(e))
        ret_sel_mask = None if sel_mask is None else gr.update()
        return ret_sel_mask

    if input_image is not None and input_image.shape == seg_image.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, seg_image, 0.5, 0)
    else:
        ret_image = seg_image

    if sel_mask is None:
        return ret_image
    else:
        if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
            return gr.update()
        else:
            return gr.update(value=ret_image)


@clear_cache_decorator
def expand_mask(input_image, sel_mask, expand_iteration=1):
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

    if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
        return gr.update()
    else:
        return gr.update(value=ret_image)


@clear_cache_decorator
def apply_mask(input_image, sel_mask):
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None

    sel_mask_image = sam_dict["mask_image"]
    sel_mask_mask = np.logical_not(sel_mask["mask"][:, :, 0:3].astype(bool)).astype(np.uint8)
    new_sel_mask = sel_mask_image * sel_mask_mask

    sam_dict["mask_image"] = new_sel_mask

    if input_image is not None and input_image.shape == new_sel_mask.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
    else:
        ret_image = new_sel_mask

    if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
        return gr.update()
    else:
        return gr.update(value=ret_image)


@clear_cache_decorator
def add_mask(input_image, sel_mask):
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None

    sel_mask_image = sam_dict["mask_image"]
    sel_mask_mask = sel_mask["mask"][:, :, 0:3].astype(bool).astype(np.uint8)
    new_sel_mask = sel_mask_image + (sel_mask_mask * np.invert(sel_mask_image, dtype=np.uint8))

    sam_dict["mask_image"] = new_sel_mask

    if input_image is not None and input_image.shape == new_sel_mask.shape:
        ret_image = cv2.addWeighted(input_image, 0.5, new_sel_mask, 0.5, 0)
    else:
        ret_image = new_sel_mask

    if sel_mask["image"].shape == ret_image.shape and np.all(sel_mask["image"] == ret_image):
        return gr.update()
    else:
        return gr.update(value=ret_image)


def auto_resize_to_pil(input_image, mask_image):
    init_image = Image.fromarray(input_image).convert("RGB")
    mask_image = Image.fromarray(mask_image).convert("RGB")
    assert init_image.size == mask_image.size, "The sizes of the image and mask do not match"
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
        if height != resize_height or width != resize_width:
            ia_logging.info(f"resize: ({height}, {width}) -> ({resize_height}, {resize_width})")
            init_image = transforms.functional.resize(init_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
            mask_image = transforms.functional.resize(mask_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS)
        if resize_height != new_height or resize_width != new_width:
            ia_logging.info(f"center_crop: ({resize_height}, {resize_width}) -> ({new_height}, {new_width})")
            init_image = transforms.functional.center_crop(init_image, (new_height, new_width))
            mask_image = transforms.functional.center_crop(mask_image, (new_height, new_width))

    return init_image, mask_image


@offload_reload_yield_decorator
@clear_cache_yield_decorator
def run_inpaint(input_image, sel_mask, prompt, n_prompt, ddim_steps, cfg_scale, seed, inp_model_id, save_mask_chk, composite_chk,
                sampler_name="DDIM", iteration_count=1):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return

    set_ia_config(IAConfig.KEYS.INP_MODEL_ID, inp_model_id, IAConfig.SECTIONS.USER)

    save_mask_image(mask_image, save_mask_chk)

    ia_logging.info(f"Loading model {inp_model_id}")
    config_offline_inpainting = get_webui_setting("inpaint_anything_offline_inpainting", False)
    if config_offline_inpainting:
        ia_logging.info("Run Inpainting on offline network: {}".format(str(config_offline_inpainting)))
    local_files_only = False
    local_file_status = download_model_from_hf(inp_model_id, local_files_only=True)
    if local_file_status != IAFileManager.DOWNLOAD_COMPLETE:
        if config_offline_inpainting:
            ia_logging.warning(local_file_status)
            return
    else:
        local_files_only = True
        ia_logging.info("local_files_only: {}".format(str(local_files_only)))

    if platform.system() == "Darwin" or devices.device == devices.cpu:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(inp_model_id, torch_dtype=torch_dtype, local_files_only=local_files_only)
    except Exception as e:
        ia_logging.error(str(e))
        if not config_offline_inpainting:
            try:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(inp_model_id, torch_dtype=torch_dtype, resume_download=True)
            except Exception as e:
                ia_logging.error(str(e))
                try:
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(inp_model_id, torch_dtype=torch_dtype, force_download=True)
                except Exception as e:
                    ia_logging.error(str(e))
                    return
        else:
            return
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

    if platform.system() == "Darwin":
        pipe = pipe.to("mps" if ia_check_versions.torch_mps_is_available else "cpu")
        pipe.enable_attention_slicing()
        torch_generator = torch.Generator(devices.cpu)
    else:
        if ia_check_versions.diffusers_enable_cpu_offload and devices.device != devices.cpu:
            ia_logging.info("Enable model cpu offload")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(devices.device)
        if shared.xformers_available:
            ia_logging.info("Enable xformers memory efficient attention")
            pipe.enable_xformers_memory_efficient_attention()
        else:
            ia_logging.info("Enable attention slicing")
            pipe.enable_attention_slicing()
        if "privateuseone" in str(getattr(devices.device, "type", "")):
            torch_generator = torch.Generator(devices.cpu)
        else:
            torch_generator = torch.Generator(devices.device)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    for count in range(int(iteration_count)):
        gc.collect()
        if seed < 0 or count > 0:
            seed = random.randint(0, 2147483647)

        generator = torch_generator.manual_seed(seed)

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
            dilate_mask_image = Image.fromarray(cv2.dilate(np.array(mask_image), np.ones((3, 3), dtype=np.uint8), iterations=4))
            output_image = Image.composite(output_image, init_image, dilate_mask_image.convert("L").filter(ImageFilter.GaussianBlur(3)))

        generation_params = {
            "Steps": ddim_steps,
            "Sampler": sampler_name,
            "CFG scale": cfg_scale,
            "Seed": seed,
            "Size": f"{width}x{height}",
            "Model": inp_model_id,
        }

        generation_params_text = ", ".join([k if k == v else f"{k}: {v}" for k, v in generation_params.items() if v is not None])
        prompt_text = prompt if prompt else ""
        negative_prompt_text = "\nNegative prompt: " + n_prompt if n_prompt else ""
        infotext = f"{prompt_text}{negative_prompt_text}\n{generation_params_text}".strip()

        metadata = PngInfo()
        metadata.add_text("parameters", infotext)

        save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(inp_model_id), str(seed)]) + ".png"
        save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
        output_image.save(save_name, pnginfo=metadata)

        yield output_image, max([1, iteration_count - (count + 1)])


@offload_reload_decorator
@clear_cache_decorator
def run_cleaner(input_image, sel_mask, cleaner_model_id, cleaner_save_mask_chk):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return None

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return None

    save_mask_image(mask_image, cleaner_save_mask_chk)

    ia_logging.info(f"Loading model {cleaner_model_id}")
    if platform.system() == "Darwin":
        model = ModelManager(name=cleaner_model_id, device=devices.cpu)
    else:
        model = ModelManager(name=cleaner_model_id, device=devices.device)

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
    output_image = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output_image)

    save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(cleaner_model_id)]) + ".png"
    save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
    output_image.save(save_name)

    del model
    return output_image


@clear_cache_decorator
def run_get_alpha_image(input_image, sel_mask):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return None, ""

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return None, ""

    alpha_image = Image.fromarray(input_image).convert("RGBA")
    mask_image = Image.fromarray(mask_image).convert("L")

    alpha_image.putalpha(mask_image)

    save_name = "_".join([ia_file_manager.savename_prefix, "rgba_image"]) + ".png"
    save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
    alpha_image.save(save_name)

    return alpha_image, f"saved: {save_name}"


@clear_cache_decorator
def run_get_mask(sel_mask):
    global sam_dict
    if sam_dict["mask_image"] is None or sel_mask is None:
        return None

    mask_image = sam_dict["mask_image"]

    save_name = "_".join([ia_file_manager.savename_prefix, "created_mask"]) + ".png"
    save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
    Image.fromarray(mask_image).save(save_name)

    return mask_image


@clear_cache_yield_decorator
def run_cn_inpaint(input_image, sel_mask,
                   cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed,
                   cn_module_id, cn_model_id, cn_save_mask_chk,
                   cn_low_vram_chk, cn_weight, cn_mode, cn_iteration_count=1,
                   cn_ref_module_id=None, cn_ref_image=None, cn_ref_weight=1.0, cn_ref_mode="Balanced", cn_ref_resize_mode="tile"):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return

    await_pre_reload_model_weights()

    if (shared.sd_model.parameterization == "v" and "sd15" in cn_model_id):
        ia_logging.error("The SDv2 model is not compatible with the ControlNet model")
        ret_image = draw_text_image(input_image, "The SD v2 model is not compatible with the ControlNet model")
        yield ret_image, 1
        return

    if (getattr(shared.sd_model, "is_sdxl", False) and "sd15" in cn_model_id):
        ia_logging.error("The SDXL model is not compatible with the ControlNet model")
        ret_image = draw_text_image(input_image, "The SD XL model is not compatible with the ControlNet model")
        yield ret_image, 1
        return

    cnet = sam_dict.get("cnet", None)
    if cnet is None:
        ia_logging.warning("The ControlNet extension is not loaded")
        return

    save_mask_image(mask_image, cn_save_mask_chk)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    input_mask = None if "inpaint_only" in cn_module_id else mask_image
    p = get_sd_img2img_processing(init_image, input_mask,
                                  cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed)

    backup_alwayson_scripts(p.scripts)
    disable_alwayson_scripts_wo_cn(cnet, p.scripts)

    cn_units = [cnet.to_processing_unit(dict(
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
    ))]

    if cn_ref_module_id is not None and cn_ref_image is not None:
        if cn_ref_resize_mode == "tile":
            ref_height, ref_width = cn_ref_image.shape[:2]
            num_h = math.ceil(height / ref_height) if height > ref_height else 1
            num_h = num_h + 1 if (num_h % 2) == 0 else num_h
            num_w = math.ceil(width / ref_width) if width > ref_width else 1
            num_w = num_w + 1 if (num_w % 2) == 0 else num_w
            cn_ref_image = np.tile(cn_ref_image, (num_h, num_w, 1))
            cn_ref_image = transforms.functional.center_crop(Image.fromarray(cn_ref_image), (height, width))
            ia_logging.info(f"Reference image is tiled ({num_h}, {num_w}) times and cropped to ({height}, {width})")
        else:
            cn_ref_image = ImageOps.fit(Image.fromarray(cn_ref_image), (width, height), method=Image.Resampling.LANCZOS)
            ia_logging.info(f"Reference image is resized and cropped to ({height}, {width})")
        assert cn_ref_image.size == init_image.size, "The sizes of the reference image and input image do not match"

        cn_units.append(cnet.to_processing_unit(dict(
            enabled=True,
            module=cn_ref_module_id,
            model=None,
            weight=cn_ref_weight,
            image={"image": np.array(cn_ref_image), "mask": None},
            resize_mode=cnet.ResizeMode.RESIZE,
            low_vram=cn_low_vram_chk,
            processor_res=min(width, height),
            guidance_start=0.0,
            guidance_end=1.0,
            pixel_perfect=True,
            control_mode=cn_ref_mode,
            threshold_a=0.5,
        )))

    p.script_args = np.zeros(get_controlnet_args_to(cnet, p.scripts)).tolist()
    cnet.update_cn_script_in_processing(p, cn_units)

    no_hash_cn_model_id = re.sub(r"\s\[[0-9a-f]{8,10}\]", "", cn_model_id).strip()

    for count in range(int(cn_iteration_count)):
        gc.collect()
        if cn_seed < 0 or count > 0:
            cn_seed = random.randint(0, 2147483647)

        p.seed = cn_seed

        try:
            processed = process_images(p)
        except devices.NansException:
            ia_logging.error("A tensor with all NaNs was produced in VAE")
            ret_image = draw_text_image(
                input_image, "A tensor with all NaNs was produced in VAE")
            clear_controlnet_cache(cnet, p.scripts)
            restore_alwayson_scripts(p.scripts)
            yield ret_image, 1
            return

        if processed is not None and len(processed.images) > 0:
            output_image = processed.images[0]

            infotext = create_infotext(p, all_prompts=p.all_prompts, all_seeds=p.all_seeds, all_subseeds=p.all_subseeds)

            metadata = PngInfo()
            metadata.add_text("parameters", infotext)

            save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(no_hash_cn_model_id), str(cn_seed)]) + ".png"
            save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
            output_image.save(save_name, pnginfo=metadata)

            yield output_image, max([1, cn_iteration_count - (count + 1)])

    clear_controlnet_cache(cnet, p.scripts)
    restore_alwayson_scripts(p.scripts)


@clear_cache_yield_decorator
def run_webui_inpaint(input_image, sel_mask,
                      webui_prompt, webui_n_prompt, webui_sampler_id, webui_ddim_steps, webui_cfg_scale, webui_strength, webui_seed,
                      webui_model_id, webui_save_mask_chk,
                      webui_mask_blur, webui_fill_mode, webui_iteration_count=1,
                      webui_enable_refiner_chk=False, webui_refiner_checkpoint="", webui_refiner_switch_at=0.8):
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        ia_logging.error("The image or mask does not exist")
        return

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("The sizes of the image and mask do not match")
        return

    if "sdxl_vae" in getattr(shared.opts, "sd_vae", ""):
        ia_logging.error("The SDXL VAE is not compatible with the inpainting model")
        ret_image = draw_text_image(
            input_image, "The SDXL VAE is not compatible with the inpainting model")
        yield ret_image, 1
        return

    set_ia_config(IAConfig.KEYS.INP_WEBUI_MODEL_ID, webui_model_id, IAConfig.SECTIONS.USER)

    save_mask_image(mask_image, webui_save_mask_chk)

    info = get_closet_checkpoint_match(webui_model_id)
    if info is None:
        ia_logging.error(f"No model found: {webui_model_id}")
        return

    await_backup_reload_ckpt_info(info=info)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    p = get_sd_img2img_processing(init_image, mask_image,
                                  webui_prompt, webui_n_prompt, webui_sampler_id, webui_ddim_steps, webui_cfg_scale, webui_strength, webui_seed,
                                  webui_mask_blur, webui_fill_mode)

    backup_alwayson_scripts(p.scripts)
    disable_all_alwayson_scripts(p.scripts)

    p.script_args = np.zeros(get_max_args_to(p.scripts)).tolist()

    if ia_check_versions.webui_refiner_is_available and webui_enable_refiner_chk:
        p.refiner_checkpoint = webui_refiner_checkpoint
        p.refiner_switch_at = webui_refiner_switch_at

    no_hash_webui_model_id = re.sub(r"\s\[[0-9a-f]{8,10}\]", "", webui_model_id).strip()
    no_hash_webui_model_id = os.path.splitext(no_hash_webui_model_id)[0]

    for count in range(int(webui_iteration_count)):
        gc.collect()
        if webui_seed < 0 or count > 0:
            webui_seed = random.randint(0, 2147483647)

        p.seed = webui_seed

        try:
            processed = process_images(p)
        except devices.NansException:
            ia_logging.error("A tensor with all NaNs was produced in VAE")
            ret_image = draw_text_image(
                input_image, "A tensor with all NaNs was produced in VAE")
            restore_alwayson_scripts(p.scripts)
            yield ret_image, 1
            return

        if processed is not None and len(processed.images) > 0:
            output_image = processed.images[0]

            infotext = create_infotext(p, all_prompts=p.all_prompts, all_seeds=p.all_seeds, all_subseeds=p.all_subseeds)

            metadata = PngInfo()
            metadata.add_text("parameters", infotext)

            save_name = "_".join([ia_file_manager.savename_prefix, os.path.basename(no_hash_webui_model_id), str(webui_seed)]) + ".png"
            save_name = os.path.join(ia_file_manager.outputs_dir, save_name)
            output_image.save(save_name, pnginfo=metadata)

            yield output_image, max([1, webui_iteration_count - (count + 1)])

    restore_alwayson_scripts(p.scripts)


def on_ui_tabs():
    global sam_dict

    setup_ia_config_ini()
    sampler_names = get_sampler_names()
    sam_model_ids = get_sam_model_ids()
    sam_model_index = get_ia_config_index(IAConfig.KEYS.SAM_MODEL_ID, IAConfig.SECTIONS.USER)
    inp_model_ids = get_inp_model_ids()
    inp_model_index = get_ia_config_index(IAConfig.KEYS.INP_MODEL_ID, IAConfig.SECTIONS.USER)
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
    cn_sampler_index = cn_sampler_ids.index("DDIM") if "DDIM" in cn_sampler_ids else 0

    cn_ref_only = False
    if cn_enabled and sam_dict["cnet"].get_max_models_num() > 1:
        cn_ref_module_ids = [cn for cn in sam_dict["cnet"].get_modules() if "reference" in cn]
        if len(cn_ref_module_ids) > 0:
            cn_ref_only = True

    webui_inpaint_enabled = False
    webui_model_ids = get_inp_webui_model_ids()
    if len(webui_model_ids) > 0:
        webui_inpaint_enabled = True
        webui_model_index = get_ia_config_index(IAConfig.KEYS.INP_WEBUI_MODEL_ID, IAConfig.SECTIONS.USER)

    if samplers_for_img2img is not None and len(samplers_for_img2img) > 0:
        webui_sampler_ids = [sampler.name for sampler in samplers_for_img2img]
    else:
        webui_sampler_ids = ["DDIM"]
    webui_sampler_index = webui_sampler_ids.index("DDIM") if "DDIM" in webui_sampler_ids else 0

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
                            status_text = gr.Textbox(label="", elem_id="status_text", max_lines=1, show_label=False, interactive=False)
                with gr.Row():
                    input_image = gr.Image(label="Input image", elem_id="ia_input_image", source="upload", type="numpy", interactive=True)

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
                    with gr.Column():
                        anime_style_chk = gr.Checkbox(label="Anime Style (Up Detection, Down mask Quality)", elem_id="anime_style_chk",
                                                      show_label=True, interactive=True)
                    with gr.Column():
                        sam_btn = gr.Button("Run Segment Anything", elem_id="sam_btn", variant="primary", interactive=False)

                with gr.Tab("Inpainting", elem_id="inpainting_tab"):
                    with gr.Row():
                        with gr.Column():
                            prompt = gr.Textbox(label="Inpainting Prompt", elem_id="ia_sd_prompt")
                            n_prompt = gr.Textbox(label="Negative Prompt", elem_id="ia_sd_n_prompt")
                        with gr.Column(scale=0, min_width=128):
                            gr.Markdown("Get prompt from:")
                            get_txt2img_prompt_btn = gr.Button("txt2img", elem_id="get_txt2img_prompt_btn")
                            get_img2img_prompt_btn = gr.Button("img2img", elem_id="get_img2img_prompt_btn")
                    with gr.Accordion("Advanced options", elem_id="inp_advanced_options", open=False):
                        composite_chk = gr.Checkbox(label="Mask area Only", elem_id="composite_chk", value=True, show_label=True, interactive=True)
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
                            inp_model_id = gr.Dropdown(label="Inpainting Model ID", elem_id="inp_model_id",
                                                       choices=inp_model_ids, value=inp_model_ids[inp_model_index], show_label=True)
                        with gr.Column():
                            with gr.Row():
                                inpaint_btn = gr.Button("Run Inpainting", elem_id="inpaint_btn", variant="primary")
                            with gr.Row():
                                save_mask_chk = gr.Checkbox(label="Save mask", elem_id="save_mask_chk",
                                                            value=False, show_label=False, interactive=False, visible=False)
                                iteration_count = gr.Slider(label="Iteration", elem_id="iteration_count", minimum=1, maximum=10, value=1, step=1)

                    with gr.Row():
                        out_image = gr.Image(label="Inpainted image", elem_id="ia_out_image", type="pil",
                                             interactive=False, show_label=False).style(height=480)

                with gr.Tab("Cleaner", elem_id="cleaner_tab"):
                    with gr.Row():
                        with gr.Column():
                            cleaner_model_id = gr.Dropdown(label="Cleaner Model ID", elem_id="cleaner_model_id",
                                                           choices=cleaner_model_ids, value=cleaner_model_ids[0], show_label=True)
                        with gr.Column():
                            with gr.Row():
                                cleaner_btn = gr.Button("Run Cleaner", elem_id="cleaner_btn", variant="primary")
                            with gr.Row():
                                cleaner_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="cleaner_save_mask_chk",
                                                                    value=False, show_label=False, interactive=False, visible=False)

                    with gr.Row():
                        cleaner_out_image = gr.Image(label="Cleaned image", elem_id="ia_cleaner_out_image", type="pil",
                                                     interactive=False, show_label=False).style(height=480)

                if webui_inpaint_enabled:
                    with gr.Tab("Inpainting webui", elem_id="webui_inpainting_tab"):
                        with gr.Row():
                            with gr.Column():
                                webui_prompt = gr.Textbox(label="Inpainting Prompt", elem_id="ia_webui_sd_prompt")
                                webui_n_prompt = gr.Textbox(label="Negative Prompt", elem_id="ia_webui_sd_n_prompt")
                            with gr.Column(scale=0, min_width=128):
                                gr.Markdown("Get prompt from:")
                                webui_get_txt2img_prompt_btn = gr.Button("txt2img", elem_id="webui_get_txt2img_prompt_btn")
                                webui_get_img2img_prompt_btn = gr.Button("img2img", elem_id="webui_get_img2img_prompt_btn")
                        with gr.Accordion("Advanced options", elem_id="webui_advanced_options", open=False):
                            webui_mask_blur = gr.Slider(label="Mask blur", minimum=0, maximum=64, step=1, value=4, elem_id="webui_mask_blur")
                            webui_fill_mode = gr.Radio(label="Masked content", elem_id="webui_fill_mode",
                                                       choices=["fill", "original", "latent noise", "latent nothing"], value="original", type="index")
                            with gr.Row():
                                with gr.Column():
                                    webui_sampler_id = gr.Dropdown(label="Sampling method webui", elem_id="webui_sampler_id",
                                                                   choices=webui_sampler_ids, value=webui_sampler_ids[webui_sampler_index], show_label=True)
                                with gr.Column():
                                    webui_ddim_steps = gr.Slider(label="Sampling steps webui", elem_id="webui_ddim_steps",
                                                                 minimum=1, maximum=150, value=30, step=1)
                            webui_cfg_scale = gr.Slider(label="Guidance scale webui", elem_id="webui_cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                            webui_strength = gr.Slider(label="Denoising strength webui", elem_id="webui_strength",
                                                       minimum=0.0, maximum=1.0, value=0.75, step=0.01)
                            webui_seed = gr.Slider(
                                label="Seed",
                                elem_id="webui_sd_seed",
                                minimum=-1,
                                maximum=2147483647,
                                step=1,
                                value=-1,
                            )
                        if ia_check_versions.webui_refiner_is_available:
                            with gr.Accordion("Refiner options", elem_id="webui_refiner_options", open=False):
                                with gr.Row():
                                    webui_enable_refiner_chk = gr.Checkbox(label="Enable Refiner", elem_id="webui_enable_refiner_chk",
                                                                           value=False, show_label=True, interactive=True)
                                with gr.Row():
                                    webui_refiner_checkpoint = gr.Dropdown(label="Refiner Model ID", elem_id="webui_refiner_checkpoint",
                                                                           choices=shared.list_checkpoint_tiles(), value="")
                                    webui_refiner_switch_at = gr.Slider(value=0.8, label="Switch at", minimum=0.01, maximum=1.0, step=0.01,
                                                                        elem_id="webui_refiner_switch_at")

                        with gr.Row():
                            with gr.Column():
                                webui_model_id = gr.Dropdown(label="Inpainting Model ID webui", elem_id="webui_model_id",
                                                             choices=webui_model_ids, value=webui_model_ids[webui_model_index], show_label=True)
                            with gr.Column():
                                with gr.Row():
                                    webui_inpaint_btn = gr.Button("Run Inpainting", elem_id="webui_inpaint_btn", variant="primary")
                                with gr.Row():
                                    webui_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="webui_save_mask_chk",
                                                                      value=False, show_label=False, interactive=False, visible=False)
                                    webui_iteration_count = gr.Slider(label="Iteration", elem_id="webui_iteration_count",
                                                                      minimum=1, maximum=10, value=1, step=1)

                        with gr.Row():
                            webui_out_image = gr.Image(label="Inpainted image", elem_id="ia_webui_out_image", type="pil",
                                                       interactive=False, show_label=False).style(height=480)

                with gr.Tab("ControlNet Inpaint", elem_id="cn_inpaint_tab"):
                    if cn_enabled:
                        with gr.Row():
                            with gr.Column():
                                cn_prompt = gr.Textbox(label="Inpainting Prompt", elem_id="ia_cn_sd_prompt")
                                cn_n_prompt = gr.Textbox(label="Negative Prompt", elem_id="ia_cn_sd_n_prompt")
                            with gr.Column(scale=0, min_width=128):
                                gr.Markdown("Get prompt from:")
                                cn_get_txt2img_prompt_btn = gr.Button("txt2img", elem_id="cn_get_txt2img_prompt_btn")
                                cn_get_img2img_prompt_btn = gr.Button("img2img", elem_id="cn_get_img2img_prompt_btn")
                        with gr.Accordion("Advanced options", elem_id="cn_advanced_options", open=False):
                            with gr.Row():
                                with gr.Column():
                                    cn_sampler_id = gr.Dropdown(label="Sampling method", elem_id="cn_sampler_id",
                                                                choices=cn_sampler_ids, value=cn_sampler_ids[cn_sampler_index], show_label=True)
                                with gr.Column():
                                    cn_ddim_steps = gr.Slider(label="Sampling steps", elem_id="cn_ddim_steps", minimum=1, maximum=150, value=30, step=1)
                            cn_cfg_scale = gr.Slider(label="Guidance scale", elem_id="cn_cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                            cn_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Denoising strength", value=0.75, elem_id="cn_strength")
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
                                    cn_low_vram_chk = gr.Checkbox(label="Low VRAM", elem_id="cn_low_vram_chk", value=True, show_label=True, interactive=True)
                                    cn_weight = gr.Slider(label="Control Weight", elem_id="cn_weight", minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                                with gr.Column():
                                    cn_mode = gr.Dropdown(label="Control Mode", elem_id="cn_mode", choices=cn_modes, value=cn_modes[-1], show_label=True)

                            if cn_ref_only:
                                with gr.Row():
                                    with gr.Column():
                                        gr.Markdown("Reference-Only Control (enabled with image below)")
                                        cn_ref_image = gr.Image(label="Reference Image", elem_id="cn_ref_image", source="upload", type="numpy",
                                                                interactive=True)
                                    with gr.Column():
                                        cn_ref_resize_mode = gr.Radio(label="Reference Image Resize Mode", elem_id="cn_ref_resize_mode",
                                                                      choices=["tile", "resize"], value="tile", show_label=True)
                                        cn_ref_module_id = gr.Dropdown(label="Reference Type", elem_id="cn_ref_module_id",
                                                                       choices=cn_ref_module_ids, value=cn_ref_module_ids[-1], show_label=True)
                                        cn_ref_weight = gr.Slider(label="Reference Control Weight", elem_id="cn_ref_weight",
                                                                  minimum=0.0, maximum=2.0, value=1.0, step=0.05)
                                        cn_ref_mode = gr.Dropdown(label="Reference Control Mode", elem_id="cn_ref_mode",
                                                                  choices=cn_modes, value=cn_modes[0], show_label=True)
                            else:
                                with gr.Row():
                                    gr.Markdown("The Multi ControlNet setting is currently set to 1.<br>"
                                                "If you wish to use the Reference-Only Control, "
                                                "please adjust the Multi ControlNet setting to 2 or more and restart the Web UI.")

                        with gr.Row():
                            with gr.Column():
                                cn_module_id = gr.Dropdown(label="ControlNet Preprocessor", elem_id="cn_module_id",
                                                           choices=cn_module_ids, value=cn_module_ids[cn_module_index], show_label=True)
                                cn_model_id = gr.Dropdown(label="ControlNet Model ID", elem_id="cn_model_id",
                                                          choices=cn_model_ids, value=cn_model_ids[0], show_label=True)
                            with gr.Column():
                                with gr.Row():
                                    cn_inpaint_btn = gr.Button("Run ControlNet Inpaint", elem_id="cn_inpaint_btn", variant="primary")
                                with gr.Row():
                                    cn_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="cn_save_mask_chk",
                                                                   value=False, show_label=False, interactive=False, visible=False)
                                    cn_iteration_count = gr.Slider(label="Iteration", elem_id="cn_iteration_count",
                                                                   minimum=1, maximum=10, value=1, step=1)

                        with gr.Row():
                            cn_out_image = gr.Image(label="Inpainted image", elem_id="ia_cn_out_image", type="pil",
                                                    interactive=False, show_label=False).style(height=480)

                    else:
                        if sam_dict["cnet"] is None:
                            gr.Markdown("ControlNet extension is not available.<br>"
                                        "Requires the [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension.")
                        elif len(cn_module_ids) > 0:
                            cn_models_directory = os.path.join("extensions", "sd-webui-controlnet", "models")
                            gr.Markdown("ControlNet inpaint model is not available.<br>"
                                        "Requires the [ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1) inpaint model "
                                        f"in the {cn_models_directory} directory.")
                        else:
                            gr.Markdown("ControlNet inpaint preprocessor is not available.<br>"
                                        "The local version of [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension may be old.")

                with gr.Tab("Mask only", elem_id="mask_only_tab"):
                    with gr.Row():
                        with gr.Column():
                            get_alpha_image_btn = gr.Button("Get mask as alpha of image", elem_id="get_alpha_image_btn")
                        with gr.Column():
                            get_mask_btn = gr.Button("Get mask", elem_id="get_mask_btn")

                    with gr.Row():
                        with gr.Column():
                            alpha_out_image = gr.Image(label="Alpha channel image", elem_id="alpha_out_image", type="pil", image_mode="RGBA", interactive=False)
                        with gr.Column():
                            mask_out_image = gr.Image(label="Mask image", elem_id="mask_out_image", type="numpy", interactive=False)

                    with gr.Row():
                        with gr.Column():
                            get_alpha_status_text = gr.Textbox(label="", elem_id="get_alpha_status_text", max_lines=1, show_label=False, interactive=False)
                        with gr.Column():
                            mask_send_to_inpaint_btn = gr.Button("Send to img2img inpaint", elem_id="mask_send_to_inpaint_btn")

            with gr.Column():
                with gr.Row():
                    gr.Markdown("Mouse over image: Press `S` key for Fullscreen mode, `R` key to Reset zoom")
                with gr.Row():
                    sam_image = gr.Image(label="Segment Anything image", elem_id="ia_sam_image", type="numpy", tool="sketch", brush_radius=8,
                                         show_label=False, interactive=True).style(height=480)
                with gr.Row():
                    with gr.Column():
                        select_btn = gr.Button("Create Mask", elem_id="select_btn", variant="primary")
                    with gr.Column():
                        with gr.Row():
                            invert_chk = gr.Checkbox(label="Invert mask", elem_id="invert_chk", show_label=True, interactive=True)
                            ignore_black_chk = gr.Checkbox(label="Ignore black area", elem_id="ignore_black_chk", value=True, show_label=True, interactive=True)

                with gr.Row():
                    sel_mask = gr.Image(label="Selected mask image", elem_id="ia_sel_mask", type="numpy", tool="sketch", brush_radius=12,
                                        show_label=False, interactive=True).style(height=480)

                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        expand_mask_btn = gr.Button("Expand mask region", elem_id="expand_mask_btn")
                    with gr.Column():
                        apply_mask_btn = gr.Button("Trim mask by sketch", elem_id="apply_mask_btn")
                        add_mask_btn = gr.Button("Add mask by sketch", elem_id="add_mask_btn")

            load_model_btn.click(download_model, inputs=[sam_model_id], outputs=[status_text])
            input_image.upload(input_image_upload, inputs=[input_image, sam_image, sel_mask], outputs=[sam_image, sel_mask, sam_btn])
            padding_btn.click(run_padding, inputs=[input_image, pad_scale_width, pad_scale_height, pad_lr_barance, pad_tb_barance, padding_mode],
                              outputs=[input_image, status_text])
            sam_btn.click(run_sam, inputs=[input_image, sam_model_id, sam_image, anime_style_chk], outputs=[sam_image, status_text]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSamMask")
            select_btn.click(select_mask, inputs=[input_image, sam_image, invert_chk, ignore_black_chk, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            expand_mask_btn.click(expand_mask, inputs=[input_image, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            apply_mask_btn.click(apply_mask, inputs=[input_image, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            add_mask_btn.click(add_mask, inputs=[input_image, sel_mask], outputs=[sel_mask]).then(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            get_txt2img_prompt_btn.click(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_getTxt2imgPrompt")
            get_img2img_prompt_btn.click(
                fn=None, inputs=None, outputs=None, _js="inpaintAnything_getImg2imgPrompt")

            inpaint_btn.click(
                run_inpaint,
                inputs=[input_image, sel_mask, prompt, n_prompt, ddim_steps, cfg_scale, seed, inp_model_id, save_mask_chk, composite_chk,
                        sampler_name, iteration_count],
                outputs=[out_image, iteration_count])
            cleaner_btn.click(
                run_cleaner,
                inputs=[input_image, sel_mask, cleaner_model_id, cleaner_save_mask_chk],
                outputs=[cleaner_out_image])
            get_alpha_image_btn.click(
                run_get_alpha_image,
                inputs=[input_image, sel_mask],
                outputs=[alpha_out_image, get_alpha_status_text])
            get_mask_btn.click(
                run_get_mask,
                inputs=[sel_mask],
                outputs=[mask_out_image])
            mask_send_to_inpaint_btn.click(
                fn=None,
                _js="inpaintAnything_sendToInpaint",
                inputs=None,
                outputs=None)
            if cn_enabled:
                cn_get_txt2img_prompt_btn.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_cnGetTxt2imgPrompt")
                cn_get_img2img_prompt_btn.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_cnGetImg2imgPrompt")
            if cn_enabled:
                cn_inputs = [input_image, sel_mask,
                             cn_prompt, cn_n_prompt, cn_sampler_id, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed,
                             cn_module_id, cn_model_id, cn_save_mask_chk,
                             cn_low_vram_chk, cn_weight, cn_mode, cn_iteration_count]
                if cn_ref_only:
                    cn_inputs.extend([cn_ref_module_id, cn_ref_image, cn_ref_weight, cn_ref_mode, cn_ref_resize_mode])
                cn_inpaint_btn.click(
                    run_cn_inpaint,
                    inputs=cn_inputs,
                    outputs=[cn_out_image, cn_iteration_count]).then(
                    fn=async_post_reload_model_weights, inputs=None, outputs=None)
            if webui_inpaint_enabled:
                webui_get_txt2img_prompt_btn.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_webuiGetTxt2imgPrompt")
                webui_get_img2img_prompt_btn.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_webuiGetImg2imgPrompt")
                wi_inputs = [input_image, sel_mask,
                             webui_prompt, webui_n_prompt, webui_sampler_id, webui_ddim_steps, webui_cfg_scale, webui_strength, webui_seed,
                             webui_model_id, webui_save_mask_chk,
                             webui_mask_blur, webui_fill_mode, webui_iteration_count]
                if ia_check_versions.webui_refiner_is_available:
                    wi_inputs.extend([webui_enable_refiner_chk, webui_refiner_checkpoint, webui_refiner_switch_at])
                webui_inpaint_btn.click(
                    run_webui_inpaint,
                    inputs=wi_inputs,
                    outputs=[webui_out_image, webui_iteration_count]).then(
                    fn=async_post_reload_model_weights, inputs=None, outputs=None)

    return [(inpaint_anything_interface, "Inpaint Anything", "inpaint_anything")]


def on_ui_settings():
    section = ("inpaint_anything", "Inpaint Anything")
    shared.opts.add_option("inpaint_anything_save_folder",
                           shared.OptionInfo(
                               default="inpaint-anything",
                               label="Folder name where output images will be saved",
                               component=gr.Radio,
                               component_args={"choices": ["inpaint-anything", "img2img-images"]},
                               section=section))
    shared.opts.add_option("inpaint_anything_sam_oncpu",
                           shared.OptionInfo(
                               default=False,
                               label="Run Segment Anything on CPU",
                               component=gr.Checkbox,
                               component_args={"interactive": True},
                               section=section))
    shared.opts.add_option("inpaint_anything_offline_inpainting",
                           shared.OptionInfo(
                               default=False,
                               label="Run Inpainting on offline network",
                               component=gr.Checkbox,
                               component_args={"interactive": True},
                               section=section))
    shared.opts.add_option("inpaint_anything_padding_fill",
                           shared.OptionInfo(
                               default=127,
                               label="Fill value used when Padding is set to constant",
                               component=gr.Slider,
                               component_args={"minimum": 0, "maximum": 255, "step": 1},
                               section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
