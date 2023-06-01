import os
import importlib
import modules.scripts as scripts
from modules import paths
from modules.shared import opts, sd_model
from modules.processing import StableDiffusionProcessingImg2Img
import copy

original_alwayson_scripts = None

def find_controlnet():
    try:
        cnet = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
    except:
        try:
            cnet = importlib.import_module('extensions-builtin.sd-webui-controlnet.scripts.external_code', 'external_code')
        except:
            cnet = None
    
    return cnet

def list_default_scripts():
    scripts_list = []

    basedir = os.path.join(paths.script_path, "scripts")
    if os.path.exists(basedir):
        for filename in sorted(os.listdir(basedir)):
            if filename.endswith(".py"):
                scripts_list.append(filename)

    return scripts_list

def backup_alwayson_scripts(input_scripts):
    global original_alwayson_scripts
    original_alwayson_scripts = copy.copy(input_scripts.alwayson_scripts)

def disable_alwayson_scripts(input_scripts):
    default_scripts = list_default_scripts()

    disabled_scripts = []
    for script in input_scripts.alwayson_scripts:
        if os.path.basename(script.filename) in default_scripts:
            continue
        if "controlnet" in os.path.basename(script.filename):
            continue
        # print("Disabled script: {}".format(os.path.basename(script.filename)))
        disabled_scripts.append(script)
    
    for script in disabled_scripts:
        input_scripts.alwayson_scripts.remove(script)

def restore_alwayson_scripts(input_scripts):
    global original_alwayson_scripts
    if original_alwayson_scripts is not None:
        input_scripts.alwayson_scripts = original_alwayson_scripts
        original_alwayson_scripts = None

def get_controlnet_args_to(input_scripts):
    for script in input_scripts.alwayson_scripts:
        if "controlnet" in os.path.basename(script.filename):
            return script.args_to
    return 1

def get_sd_img2img_processing(init_image, mask_image, prompt, n_prompt, sampler_id, ddim_steps, cfg_scale, strength, seed):
    width, height = init_image.size

    sd_img2img_args = dict(
        sd_model=sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        inpaint_full_res=False,
        init_images=[init_image],
        resize_mode=0,  # 0:Just resize
        denoising_strength=strength,
        image_cfg_scale=1.5,
        mask=mask_image,
        mask_blur=4,
        inpainting_fill=1,  # 1:original
        inpainting_mask_invert=0,   # 0:Inpaint masked
        prompt=prompt,
        negative_prompt=n_prompt,
        seed=seed,
        sampler_name=sampler_id,
        batch_size=1,
        n_iter=1,
        steps=ddim_steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=True,
    )

    p = StableDiffusionProcessingImg2Img(**sd_img2img_args)

    p.is_img2img = True
    p.scripts = scripts.scripts_img2img

    return p
