import os
import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, UniPCMultistepScheduler
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from scripts.get_dataset_colormap import create_pascal_label_colormap
from scripts.webui_controlnet import find_controlnet
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

from modules.processing import StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, sd_model
from modules.sd_samplers import samplers_for_img2img

def get_sam_model_ids():
    """Get SAM model ids list.

    Returns:
        list: SAM model ids list
    """
    sam_model_ids = [
        "sam_vit_h_4b8939.pth",
        "sam_vit_l_0b3195.pth",
        "sam_vit_b_01ec64.pth",
        ]
    return sam_model_ids

def download_model(sam_model_id):
    """Download SAM model.

    Args:
        sam_model_id (str): SAM model id

    Returns:
        str: download status
    """
    # print(sam_model_id)
    # url_sam_vit_h_4b8939 = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    url_sam = "https://dl.fbaipublicfiles.com/segment_anything/" + sam_model_id
    models_dir = os.path.join(extensions_dir, "sd-webui-inpaint-anything", "models")
    sam_checkpoint = os.path.join(models_dir, sam_model_id)
    if not os.path.isfile(sam_checkpoint):
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        
        download_url_to_file(url_sam, sam_checkpoint)
        
        return "Download complete"
    else:
        return "Model already exists"

def get_sam_mask_generator(sam_checkpoint):
    """Get SAM mask generator.

    Args:
        sam_checkpoint (str): SAM checkpoint path

    Returns:
        SamAutomaticMaskGenerator or None: SAM mask generator
    """
    # model_type = "vit_h"
    model_type = os.path.basename(sam_checkpoint)[4:9]

    if os.path.isfile(sam_checkpoint):
        torch.load = unsafe_torch_load
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_mask_generator = SamAutomaticMaskGenerator(sam)
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
    model_type = os.path.basename(sam_checkpoint)[4:9]

    if os.path.isfile(sam_checkpoint):
        torch.load = unsafe_torch_load
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
        torch.load = load
    else:
        sam_predictor = None
    
    return sam_predictor

ia_outputs_dir = os.path.join(os.path.dirname(extensions_dir),
                          "outputs", "inpaint-anything",
                          datetime.now().strftime("%Y-%m-%d"))

sam_dict = dict(sam_masks=None, mask_image=None, cnet=None)

def get_model_ids():
    """Get inpainting model ids list.

    Returns:
        list: model ids list
    """
    model_ids = [
        "stabilityai/stable-diffusion-2-inpainting",
        "Uminosachi/dreamshaper_5-inpainting",
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

def clear_cache():
    gc.collect()
    torch_gc()

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
    print("input_image:", input_image.shape, input_image.dtype)
    
    cm_pascal = create_pascal_label_colormap()
    seg_colormap = cm_pascal
    seg_colormap = [c for c in seg_colormap if max(c) >= 64]
    # print(len(seg_colormap))
    
    sam_mask_generator = get_sam_mask_generator(sam_checkpoint)
    print(f"{sam_mask_generator.__class__.__name__} {sam_model_id}")
    sam_masks = sam_mask_generator.generate(input_image)

    canvas_image = np.zeros_like(input_image)

    sam_masks = sorted(sam_masks, key=lambda x: np.sum(x.get("segmentation").astype(int)))
    sam_masks = sam_masks[:len(seg_colormap)]
    for idx, seg_dict in enumerate(sam_masks):
        seg_mask = np.expand_dims(seg_dict.get("segmentation").astype(int), axis=-1)
        canvas_mask = np.logical_not(np.sum(canvas_image, axis=-1, keepdims=True).astype(bool)).astype(int)
        seg_color = seg_colormap[idx] * seg_mask * canvas_mask
        canvas_image = canvas_image + seg_color
    seg_image = canvas_image.astype(np.uint8)
    
    sam_dict["sam_masks"] = sam_masks

    clear_cache()
    if sam_image is None:
        return seg_image, "Segment Anything completed"
    else:
        if sam_image["image"].shape == seg_image.shape and np.all(sam_image["image"] == seg_image):
            return gr.update(), "Segment Anything completed"
        else:
            return gr.update(value=seg_image), "Segment Anything completed"

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
    
    for i in range(expand_iteration):
        new_sel_mask = np.array(cv2.dilate(new_sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=1))
    
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
    # print(init_image.size, mask_image.size)
    width, height = init_image.size

    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    if new_width < width or new_height < height:
        if (new_width / width) < (new_height / height):
            scale = new_height / height
        else:
            scale = new_width / width
        print("resize:", f"({height}, {width})", "->", f"({int(height*scale+0.5)}, {int(width*scale+0.5)})")
        init_image = transforms.functional.resize(init_image, (int(height*scale+0.5), int(width*scale+0.5)), transforms.InterpolationMode.LANCZOS)
        mask_image = transforms.functional.resize(mask_image, (int(height*scale+0.5), int(width*scale+0.5)), transforms.InterpolationMode.LANCZOS)
        print("center_crop:", f"({int(height*scale+0.5)}, {int(width*scale+0.5)})", "->", f"({new_height}, {new_width})")
        init_image = transforms.functional.center_crop(init_image, (new_height, new_width))
        mask_image = transforms.functional.center_crop(mask_image, (new_height, new_width))
        assert init_image.size == mask_image.size, "The size of image and mask do not match"
    
    return init_image, mask_image

def run_inpaint(input_image, sel_mask, prompt, n_prompt, ddim_steps, cfg_scale, seed, model_id, save_mask_chk):
    clear_cache()
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        return None

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        print("The size of image and mask do not match")
        return None

    global ia_outputs_dir
    config_save_folder = shared.opts.data.get("inpaint_anything_save_folder", "inpaint-anything")
    if config_save_folder in ["inpaint-anything", "img2img-images"]:
        ia_outputs_dir = os.path.join(os.path.dirname(extensions_dir),
                                      "outputs", config_save_folder,
                                      datetime.now().strftime("%Y-%m-%d"))
    if save_mask_chk:
        if not os.path.isdir(ia_outputs_dir):
            os.makedirs(ia_outputs_dir, exist_ok=True)
        save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "created_mask" + ".png"
        save_name = os.path.join(ia_outputs_dir, save_name)
        Image.fromarray(mask_image).save(save_name)

    print(model_id)
    if platform.system() == "Darwin":
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.safety_checker = None

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    if seed < 0:
        seed = random.randint(0, 2147483647)
    
    if platform.system() == "Darwin":
        pipe = pipe.to("mps")
        pipe.enable_attention_slicing()
        generator = torch.Generator("cpu").manual_seed(seed)
    else:
        # pipe.enable_model_cpu_offload()
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
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
    
    clear_cache()
    return output_image

def run_cleaner(input_image, sel_mask, cleaner_model_id, cleaner_save_mask_chk):
    clear_cache()
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        return None
    
    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        print("The size of image and mask do not match")
        return None

    global ia_outputs_dir
    config_save_folder = shared.opts.data.get("inpaint_anything_save_folder", "inpaint-anything")
    if config_save_folder in ["inpaint-anything", "img2img-images"]:
        ia_outputs_dir = os.path.join(os.path.dirname(extensions_dir),
                                      "outputs", config_save_folder,
                                      datetime.now().strftime("%Y-%m-%d"))
    if cleaner_save_mask_chk:
        if not os.path.isdir(ia_outputs_dir):
            os.makedirs(ia_outputs_dir, exist_ok=True)
        save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "created_mask" + ".png"
        save_name = os.path.join(ia_outputs_dir, save_name)
        Image.fromarray(mask_image).save(save_name)

    print(cleaner_model_id)
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
    
    clear_cache()
    return output_image

def run_cn_inpaint(input_image, sel_mask, cn_prompt, cn_n_prompt, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed, cn_module_id, cn_model_id, cn_save_mask_chk):
    clear_cache()
    global sam_dict
    if input_image is None or sam_dict["mask_image"] is None or sel_mask is None:
        return None

    mask_image = sam_dict["mask_image"]
    if input_image.shape != mask_image.shape:
        print("The size of image and mask do not match")
        return None

    global ia_outputs_dir
    if cn_save_mask_chk:
        if not os.path.isdir(ia_outputs_dir):
            os.makedirs(ia_outputs_dir, exist_ok=True)
        save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "created_mask" + ".png"
        save_name = os.path.join(ia_outputs_dir, save_name)
        Image.fromarray(mask_image).save(save_name)

    print(cn_model_id)

    if cn_seed < 0:
        cn_seed = random.randint(0, 2147483647)
    
    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size
    
    p = StableDiffusionProcessingImg2Img(
        sd_model=sd_model,
        outpath_samples = opts.outdir_samples or opts.outdir_img2img_samples,
    )
    
    p.is_img2img = True
    p.scripts = scripts.scripts_img2img

    p.init_images = [init_image]
    p.image_mask = mask_image
    p.width, p.height = (width, height)
    p.prompt = cn_prompt
    p.negative_prompt = cn_n_prompt
    p.denoising_strength = cn_strength
    p.steps = cn_ddim_steps
    p.seed = cn_seed
    p.cfg_scale = cn_cfg_scale
    p.sampler_name = "DDIM"
    p.batch_size = 1
    p.do_not_save_samples = True

    cnet = sam_dict.get("cnet", None)
    if cnet is not None:
        cn_units = [cnet.ControlNetUnit(
            enabled=True,
            module=cn_module_id,
            model=cn_model_id,
            weight=1.0,
            image={"image": np.array(init_image), "mask": np.array(mask_image)},
            resize_mode=cnet.ResizeMode.INNER_FIT,
            low_vram=False,
            processor_res=512,
            threshold_a=64,
            threshold_b=64,
            guidance_start=0.0,
            guidance_end=1.0,
            pixel_perfect=True,
            control_mode=cnet.ControlMode.BALANCED,
        )]
        
        p.script_args = {"enabled": True}
        cnet.update_cn_script_in_processing(p, cn_units, is_img2img=True, is_ui=False)

    processed = process_images(p)
    
    if processed is not None:
        if len(processed.images) > 0:
            results = processed.images[0]
        else:
            results = None
    else:
        results = None

    return results

class Script(scripts.Script):
  def __init__(self) -> None:
    super().__init__()

  def title(self):
    return "Inpaint Anything"

  def show(self, is_img2img):
    return scripts.AlwaysVisible

  def ui(self, is_img2img):
    return ()

def on_ui_tabs():
    global sam_dict
    
    sam_model_ids = get_sam_model_ids()
    model_ids = get_model_ids()
    cleaner_model_ids = get_cleaner_model_ids()
    sam_dict["cnet"] = find_controlnet() 
    if sam_dict["cnet"] is not None:
        cn_module_ids = [cn for cn in sam_dict["cnet"].get_modules() if "inpaint" in cn]
        cn_model_ids = [cn for cn in sam_dict["cnet"].get_models() if "inpaint" in cn]
    
    with gr.Blocks(analytics_enabled=False) as inpaint_anything_interface:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        sam_model_id = gr.Dropdown(label="Segment Anything Model ID", elem_id="sam_model_id", choices=sam_model_ids,
                                                   value=sam_model_ids[1], show_label=True)
                    with gr.Column():
                        with gr.Row():
                            load_model_btn = gr.Button("Download model", elem_id="load_model_btn")
                        with gr.Row():
                            status_text = gr.Textbox(label="", max_lines=1, show_label=False, interactive=False)
                input_image = gr.Image(label="Input image", elem_id="input_image", source="upload", type="numpy", interactive=True)
                sam_btn = gr.Button("Run Segment Anything", elem_id="sam_btn")
                
                with gr.Tab("Inpainting"):
                    prompt = gr.Textbox(label="Inpainting prompt", elem_id="sd_prompt")
                    n_prompt = gr.Textbox(label="Negative prompt", elem_id="sd_n_prompt")
                    with gr.Accordion("Advanced options", open=False):
                        ddim_steps = gr.Slider(label="Sampling Steps", elem_id="ddim_steps", minimum=1, maximum=50, value=20, step=1)
                        cfg_scale = gr.Slider(label="Guidance Scale", elem_id="cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                        seed = gr.Slider(
                            label="Seed",
                            elem_id="sd_seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1,
                            # randomize=True,
                        )
                    with gr.Row():
                        with gr.Column():
                            model_id = gr.Dropdown(label="Inpainting Model ID", elem_id="model_id", choices=model_ids, value=model_ids[0], show_label=True)
                        with gr.Column():
                            with gr.Row():
                                inpaint_btn = gr.Button("Run Inpainting", elem_id="inpaint_btn")
                            with gr.Row():
                                save_mask_chk = gr.Checkbox(label="Save mask", elem_id="save_mask_chk", show_label=True, interactive=True)

                    out_image = gr.Image(label="Inpainted image", elem_id="out_image", interactive=False).style(height=480)
                
                with gr.Tab("Cleaner"):
                    with gr.Row():
                        with gr.Column():
                            cleaner_model_id = gr.Dropdown(label="Cleaner Model ID", elem_id="cleaner_model_id", choices=cleaner_model_ids, value=cleaner_model_ids[0], show_label=True)
                        with gr.Column():
                            with gr.Row():
                                cleaner_btn = gr.Button("Run Cleaner", elem_id="cleaner_btn")
                            with gr.Row():
                                cleaner_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="cleaner_save_mask_chk", show_label=True, interactive=True)
                    
                    cleaner_out_image = gr.Image(label="Cleaned image", elem_id="cleaner_out_image", interactive=False).style(height=480)

                with gr.Tab("ControlNet Inpainting"):
                    if sam_dict.get("cnet", None) is not None and len(cn_module_ids) > 0 and len(cn_model_ids) > 0:
                        cn_prompt = gr.Textbox(label="Inpainting prompt", elem_id="cn_sd_prompt")
                        cn_n_prompt = gr.Textbox(label="Negative prompt", elem_id="cn_sd_n_prompt")
                        with gr.Accordion("Advanced options", open=False):
                            cn_ddim_steps = gr.Slider(label="Sampling Steps", elem_id="cn_ddim_steps", minimum=1, maximum=50, value=20, step=1)
                            cn_cfg_scale = gr.Slider(label="Guidance Scale", elem_id="cn_cfg_scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                            cn_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoising Strength', value=0.75, elem_id="cn_strength")
                            cn_seed = gr.Slider(
                                label="Seed",
                                elem_id="cn_sd_seed",
                                minimum=-1,
                                maximum=2147483647,
                                step=1,
                                value=-1,
                                # randomize=True,
                            )
                        with gr.Row():
                            with gr.Column():
                                cn_module_id = gr.Dropdown(label="ControlNet Preprocessor", elem_id="cn_module_id", choices=cn_module_ids, value=cn_module_ids[0], show_label=True)
                                cn_model_id = gr.Dropdown(label="ControlNet Model ID", elem_id="cn_model_id", choices=cn_model_ids, value=cn_model_ids[0], show_label=True)
                            with gr.Column():
                                with gr.Row():
                                    cn_inpaint_btn = gr.Button("Run Inpainting", elem_id="cn_inpaint_btn")
                                with gr.Row():
                                    cn_save_mask_chk = gr.Checkbox(label="Save mask", elem_id="cn_save_mask_chk", show_label=True, interactive=True)
                        
                        cn_out_image = gr.Image(label="Inpainted image", elem_id="cn_out_image", interactive=False).style(height=480)
                    else:
                        if sam_dict.get("cnet", None) is None:
                            gr.Markdown("ControlNet is not available.")
                        else:
                            gr.Markdown("ControlNet inpaint model is not available.")

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
                        # expand_iteration = gr.Slider(label="Iterations", elem_id="expand_iteration", minimum=1, maximum=5, value=1,
                        #                              step=1, visible=False)
                        apply_mask_btn = gr.Button("Apply sketch to mask", elem_id="apply_mask_btn")
            
            load_model_btn.click(download_model, inputs=[sam_model_id], outputs=[status_text])
            sam_btn.click(run_sam, inputs=[input_image, sam_model_id, sam_image], outputs=[sam_image, status_text])
            select_btn.click(select_mask, inputs=[input_image, sam_image, invert_chk, sel_mask], outputs=[sel_mask])
            expand_mask_btn.click(expand_mask, inputs=[input_image, sel_mask], outputs=[sel_mask])
            apply_mask_btn.click(apply_mask, inputs=[input_image, sel_mask], outputs=[sel_mask])
            inpaint_btn.click(
                run_inpaint,
                inputs=[input_image, sel_mask, prompt, n_prompt, ddim_steps, cfg_scale, seed, model_id, save_mask_chk],
                outputs=[out_image])
            cleaner_btn.click(
                run_cleaner,
                inputs=[input_image, sel_mask, cleaner_model_id, cleaner_save_mask_chk],
                outputs=[cleaner_out_image])
            cn_inpaint_btn.click(
                run_cn_inpaint,
                inputs=[input_image, sel_mask, cn_prompt, cn_n_prompt, cn_ddim_steps, cn_cfg_scale, cn_strength, cn_seed, cn_module_id, cn_model_id, cn_save_mask_chk],
                outputs=[cn_out_image])
    
    return [(inpaint_anything_interface, "Inpaint Anything", "inpaint_anything")]

def on_ui_settings():
    section = ("inpaint_anything", "Inpaint Anything")
    shared.opts.add_option("inpaint_anything_save_folder", shared.OptionInfo(
        "inpaint-anything", "Folder name where output images will be saved", gr.Radio, {"choices": ["inpaint-anything", "img2img-images"]}, section=section))

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
