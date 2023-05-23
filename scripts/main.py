import os
import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler, UniPCMultistepScheduler
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from scripts.get_dataset_colormap import create_pascal_label_colormap
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
print("platform:", platform.system())

import modules.scripts as scripts
from modules import shared, script_callbacks
try:
    from modules.paths_internal import extensions_dir
except Exception:
    from modules.extensions import extensions_dir
from modules.devices import device, torch_gc
from modules.safe import unsafe_torch_load, load

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

sam_dict = {"sam_masks": None}

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
    # print(sam_dict)    
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
        image = sam_image["image"]
        
        if np.all(image == seg_image):
            return gr.update(), "Segment Anything completed"
        else:
            return gr.update(value=seg_image), "Segment Anything completed"

def select_mask(masks_image, invert_chk, sel_mask):
    global sam_dict
    if sam_dict["sam_masks"] is None or masks_image is None:
        clear_cache()
        return None
    sam_masks = sam_dict["sam_masks"]
    
    image = masks_image["image"]
    mask = masks_image["mask"][:,:,0:3]
    
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

    clear_cache()
    if sel_mask is None:
        return seg_image
    else:
        sel_mask_image = sel_mask["image"]
        
        if np.all(sel_mask_image == seg_image):
            return gr.update()
        else:
            return gr.update(value=seg_image)

def expand_mask(sel_mask, expand_iteration):
    if sel_mask is None:
        clear_cache()
        return None
    
    sel_mask_image = sel_mask["image"]
    sel_mask_mask = np.logical_not(sel_mask["mask"][:,:,0:3].astype(bool)).astype(np.uint8)
    sel_mask = sel_mask_image * sel_mask_mask
    
    expand_iteration = int(np.clip(expand_iteration, 1, 5))
    
    for i in range(expand_iteration):
        sel_mask = np.array(cv2.dilate(sel_mask, np.ones((3, 3), dtype=np.uint8), iterations=1))
    
    return gr.update(value=sel_mask)

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
    if input_image is None or sel_mask is None:
        clear_cache()
        return None

    sel_mask_image = sel_mask["image"]
    sel_mask_mask = np.logical_not(sel_mask["mask"][:,:,0:3].astype(bool)).astype(np.uint8)
    sel_mask = sel_mask_image * sel_mask_mask

    global ia_outputs_dir
    if save_mask_chk:
        if not os.path.isdir(ia_outputs_dir):
            os.makedirs(ia_outputs_dir, exist_ok=True)
        save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "created_mask" + ".png"
        save_name = os.path.join(ia_outputs_dir, save_name)
        Image.fromarray(sel_mask).save(save_name)

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
    
    mask_image = sel_mask
        
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
    
    if True:
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
    if input_image is None or sel_mask is None:
        clear_cache()
        return None

    sel_mask_image = sel_mask["image"]
    sel_mask_mask = np.logical_not(sel_mask["mask"][:,:,0:3].astype(bool)).astype(np.uint8)
    sel_mask = sel_mask_image * sel_mask_mask

    global ia_outputs_dir
    if cleaner_save_mask_chk:
        if not os.path.isdir(ia_outputs_dir):
            os.makedirs(ia_outputs_dir, exist_ok=True)
        save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + "created_mask" + ".png"
        save_name = os.path.join(ia_outputs_dir, save_name)
        Image.fromarray(sel_mask).save(save_name)

    print(cleaner_model_id)
    model = ModelManager(name=cleaner_model_id, device=device)

    mask_image = sel_mask
    
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
    
    if True:
        if not os.path.isdir(ia_outputs_dir):
            os.makedirs(ia_outputs_dir, exist_ok=True)
        save_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + os.path.basename(cleaner_model_id) + ".png"
        save_name = os.path.join(ia_outputs_dir, save_name)
        output_image.save(save_name)
    
    clear_cache()
    return output_image

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
    sam_model_ids = get_sam_model_ids()
    model_ids = get_model_ids()
    cleaner_model_ids = get_cleaner_model_ids()
    
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
                        expand_iteration = gr.Slider(label="Iterations", elem_id="expand_iteration", minimum=1, maximum=5, value=1,
                                                     step=1, visible=False)
            
            load_model_btn.click(download_model, inputs=[sam_model_id], outputs=[status_text])
            sam_btn.click(run_sam, inputs=[input_image, sam_model_id, sam_image], outputs=[sam_image, status_text])
            select_btn.click(select_mask, inputs=[sam_image, invert_chk, sel_mask], outputs=[sel_mask])
            expand_mask_btn.click(expand_mask, inputs=[sel_mask, expand_iteration], outputs=[sel_mask])
            inpaint_btn.click(run_inpaint, inputs=[input_image, sel_mask, prompt, n_prompt, ddim_steps, cfg_scale, seed, model_id, save_mask_chk],
                              outputs=[out_image])
            cleaner_btn.click(run_cleaner, inputs=[input_image, sel_mask, cleaner_model_id, cleaner_save_mask_chk], outputs=[cleaner_out_image])
    
    return [(inpaint_anything_interface, "Inpaint Anything", "inpaint_anything")]

script_callbacks.on_ui_tabs(on_ui_tabs)
