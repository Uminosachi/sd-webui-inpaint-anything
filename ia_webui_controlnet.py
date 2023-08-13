import copy
import importlib
import os

import modules.scripts as scripts
from modules import paths, shared
from modules.processing import StableDiffusionProcessingImg2Img

original_alwayson_scripts = None


def find_controlnet():
    """Find ControlNet external_code

    Returns:
        module: ControlNet external_code module
    """
    try:
        cnet = importlib.import_module("extensions.sd-webui-controlnet.scripts.external_code")
    except Exception:
        try:
            cnet = importlib.import_module("extensions-builtin.sd-webui-controlnet.scripts.external_code")
        except Exception:
            cnet = None

    return cnet


def list_default_scripts():
    """Get list of default scripts

    Returns:
        list: List of default scripts
    """
    scripts_list = []

    basedir = os.path.join(paths.script_path, "scripts")
    if os.path.isdir(basedir):
        for filename in sorted(os.listdir(basedir)):
            if filename.endswith(".py"):
                scripts_list.append(filename)

    # basedir = os.path.join(paths.script_path, "modules", "processing_scripts")
    # if os.path.isdir(basedir):
    #     for filename in sorted(os.listdir(basedir)):
    #         if filename.endswith(".py"):
    #             scripts_list.append(filename)

    return scripts_list


def backup_alwayson_scripts(input_scripts):
    """Backup alwayson scripts

    Args:
        input_scripts (ScriptRunner): scripts to backup alwayson scripts
    """
    global original_alwayson_scripts
    original_alwayson_scripts = copy.copy(input_scripts.alwayson_scripts)


def disable_alwayson_scripts_wo_cn(cnet, input_scripts):
    """Disable alwayson scripts

    Args:
        input_scripts (ScriptRunner): scripts to disable alwayson scripts
    """
    default_scripts = list_default_scripts()

    disabled_scripts = []
    for script in input_scripts.alwayson_scripts:
        if os.path.basename(script.filename) in default_scripts:
            continue
        if cnet.is_cn_script(script):
            continue
        # print("Disabled script: {}".format(script.title()))
        disabled_scripts.append(script)

    for script in disabled_scripts:
        input_scripts.alwayson_scripts.remove(script)


def disable_all_alwayson_scripts(input_scripts):
    """Disable all alwayson scripts

    Args:
        input_scripts (ScriptRunner): scripts to disable alwayson scripts
    """
    default_scripts = list_default_scripts()

    disabled_scripts = []
    for script in input_scripts.alwayson_scripts:
        if os.path.basename(script.filename) in default_scripts:
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


def get_controlnet_args_to(cnet, input_scripts):
    """Get args_to of ControlNet script

    Args:
        input_scripts (ScriptRunner): scripts to get args_to of ControlNet script

    Returns:
        int: args_to of ControlNet script
    """
    for script in input_scripts.alwayson_scripts:
        if cnet.is_cn_script(script):
            return script.args_to
    return get_max_args_to(input_scripts)


def clear_controlnet_cache(cnet, input_scripts):
    """Clear ControlNet cache

    Args:
        input_scripts (ScriptRunner): scripts to clear ControlNet cache
    """
    for script in input_scripts.alwayson_scripts:
        if cnet.is_cn_script(script):
            if hasattr(script, "clear_control_model_cache"):
                # print("Clear ControlNet cache: {}".format(script.title()))
                script.clear_control_model_cache()


def get_sd_img2img_processing(init_image, mask_image, prompt, n_prompt, sampler_id, ddim_steps, cfg_scale, strength, seed,
                              mask_blur=4, fill_mode=1):
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
        mask_blur (int, optional): Mask blur. Defaults to 4.
        fill_mode (int, optional): Fill mode. Defaults to 1.

    Returns:
        StableDiffusionProcessingImg2Img: StableDiffusionProcessingImg2Img instance
    """
    width, height = init_image.size

    sd_img2img_args = dict(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_img2img_samples,
        inpaint_full_res=False,
        init_images=[init_image],
        resize_mode=0,  # 0:Just resize
        denoising_strength=strength,
        image_cfg_scale=1.5,
        mask=mask_image,
        mask_blur=mask_blur,
        inpainting_fill=fill_mode,
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
