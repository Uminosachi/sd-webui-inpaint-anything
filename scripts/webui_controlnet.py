import os
import importlib
import modules.scripts as scripts
from modules import paths
from modules.shared import opts, sd_model
from modules.processing import StableDiffusionProcessingImg2Img
import copy

original_alwayson_scripts = None

def find_controlnet():
    """Find ControlNet external_code

    Returns:
        module: ControlNet external_code module
    """
    try:
        cnet = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
    except:
        try:
            cnet = importlib.import_module('extensions-builtin.sd-webui-controlnet.scripts.external_code', 'external_code')
        except:
            cnet = None
    
    return cnet

def list_default_scripts():
    """Get list of default scripts

    Returns:
        list: List of default scripts
    """
    scripts_list = []

    basedir = os.path.join(paths.script_path, "scripts")
    if os.path.exists(basedir):
        for filename in sorted(os.listdir(basedir)):
            if filename.endswith(".py"):
                scripts_list.append(filename)

    return scripts_list

def backup_alwayson_scripts(input_scripts):
    """Backup alwayson scripts

    Args:
        input_scripts (ScriptRunner): scripts to backup alwayson scripts
    """
    global original_alwayson_scripts
    original_alwayson_scripts = copy.copy(input_scripts.alwayson_scripts)

def disable_alwayson_scripts(input_scripts):
    """Disable alwayson scripts
    
    Args:
        input_scripts (ScriptRunner): scripts to disable alwayson scripts
    """
    default_scripts = list_default_scripts()

    disabled_scripts = []
    for script in input_scripts.alwayson_scripts:
        if os.path.basename(script.filename) in default_scripts:
            continue
        if "controlnet" in os.path.basename(script.filename) or script.title().lower() == "controlnet":
            continue
        # print("Disabled script: {}".format(script.title()))
        disabled_scripts.append(script)
    
    for script in disabled_scripts:
        input_scripts.alwayson_scripts.remove(script)

def restore_alwayson_scripts(input_scripts):
    """Restore alwayson scripts
    
    Args:
        input_scripts (ScriptRunner): scripts to restore alwayson scripts
    """
    global original_alwayson_scripts
    if original_alwayson_scripts is not None:
        input_scripts.alwayson_scripts = original_alwayson_scripts
        original_alwayson_scripts = None

def get_max_args_to(input_scripts):
    """Get max args_to of scripts
    
    Args:
        input_scripts (ScriptRunner): scripts to get max args_to of scripts
    
    Returns:
        int: max args_to of scripts
    """
    max_args_to = 0
    for script in input_scripts.alwayson_scripts:
        if max_args_to < script.args_to:
            max_args_to = script.args_to
    return max_args_to

def get_controlnet_args_to(input_scripts):
    """Get args_to of ControlNet script

    Args:
        input_scripts (ScriptRunner): scripts to get args_to of ControlNet script

    Returns:
        int: args_to of ControlNet script
    """
    for script in input_scripts.alwayson_scripts:
        if "controlnet" in os.path.basename(script.filename) or script.title().lower() == "controlnet":
            return script.args_to
    return get_max_args_to(input_scripts)

def clear_controlnet_cache(input_scripts):
    """Clear ControlNet cache

    Args:
        input_scripts (ScriptRunner): scripts to clear ControlNet cache
    """
    for script in input_scripts.alwayson_scripts:
        if "controlnet" in os.path.basename(script.filename) or script.title().lower() == "controlnet":
            if hasattr(script, "model_cache"):
                # print("Clear ControlNet cache: {}".format(script.title()))
                script.model_cache.clear()

def get_sd_img2img_processing(init_image, mask_image, prompt, n_prompt, sampler_id, ddim_steps, cfg_scale, strength, seed):
    """Get StableDiffusionProcessingImg2Img instance
    
    Args:
        init_image (PIL.Image): Initial image
        mask_image (PIL.Image): Mask image
        prompt (str): Prompt
        n_prompt (int): Negative prompt
        sampler_id (str): Sampler ID
        ddim_steps (int): Steps
        cfg_scale (float): CFG scale
        strength (float): Denoising strength
        seed (int): Seed

    Returns:
        StableDiffusionProcessingImg2Img: StableDiffusionProcessingImg2Img instance
    """
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
