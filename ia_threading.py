import gc
import inspect
import threading
from contextlib import ContextDecorator
from functools import wraps

import torch
from modules import devices, safe, shared
from modules.sd_models import load_model, reload_model_weights

backup_sd_model, backup_device, backup_ckpt_info = None, None, None
model_access_sem = threading.Semaphore(1)


def clear_cache():
    gc.collect()
    devices.torch_gc()


def is_sdxl_lowvram(sd_model):
    return (shared.cmd_opts.lowvram or shared.cmd_opts.medvram or
            getattr(shared.cmd_opts, "medvram_sdxl", False) and hasattr(sd_model, "conditioner"))


def webui_reload_model_weights(sd_model=None, info=None):
    try:
        reload_model_weights(sd_model=sd_model, info=info)
    except Exception:
        load_model(checkpoint_info=info)


def pre_offload_model_weights(sem):
    global backup_sd_model, backup_device, backup_ckpt_info
    with sem:
        if (shared.sd_model is not None and not is_sdxl_lowvram(shared.sd_model) and
                getattr(shared.sd_model, "device", devices.cpu) != devices.cpu):
            backup_sd_model = shared.sd_model
            backup_device = getattr(backup_sd_model, "device")
            backup_sd_model.to(devices.cpu)
            clear_cache()


def await_pre_offload_model_weights():
    global model_access_sem
    thread = threading.Thread(target=pre_offload_model_weights, args=(model_access_sem,))
    thread.start()
    thread.join()


def pre_reload_model_weights(sem):
    global backup_sd_model, backup_device, backup_ckpt_info
    with sem:
        if backup_sd_model is not None and backup_device is not None:
            backup_sd_model.to(backup_device)
            backup_sd_model, backup_device = None, None
        if shared.sd_model is not None and backup_ckpt_info is not None:
            webui_reload_model_weights(sd_model=shared.sd_model, info=backup_ckpt_info)
            backup_ckpt_info = None


def await_pre_reload_model_weights():
    global model_access_sem
    thread = threading.Thread(target=pre_reload_model_weights, args=(model_access_sem,))
    thread.start()
    thread.join()


def backup_reload_ckpt_info(sem, info):
    global backup_sd_model, backup_device, backup_ckpt_info
    with sem:
        if backup_sd_model is not None and backup_device is not None:
            backup_sd_model.to(backup_device)
            backup_sd_model, backup_device = None, None
        if shared.sd_model is not None:
            backup_ckpt_info = shared.sd_model.sd_checkpoint_info
            webui_reload_model_weights(sd_model=shared.sd_model, info=info)


def await_backup_reload_ckpt_info(info):
    global model_access_sem
    thread = threading.Thread(target=backup_reload_ckpt_info, args=(model_access_sem, info))
    thread.start()
    thread.join()


def post_reload_model_weights(sem):
    global backup_sd_model, backup_device, backup_ckpt_info
    with sem:
        if backup_sd_model is not None and backup_device is not None:
            backup_sd_model.to(backup_device)
            backup_sd_model, backup_device = None, None
        if shared.sd_model is not None and backup_ckpt_info is not None:
            webui_reload_model_weights(sd_model=shared.sd_model, info=backup_ckpt_info)
            backup_ckpt_info = None


def async_post_reload_model_weights():
    global model_access_sem
    thread = threading.Thread(target=post_reload_model_weights, args=(model_access_sem,))
    thread.start()


def acquire_release_semaphore(sem):
    with sem:
        pass


def await_acquire_release_semaphore():
    global model_access_sem
    thread = threading.Thread(target=acquire_release_semaphore, args=(model_access_sem,))
    thread.start()
    thread.join()


def clear_cache_decorator(func):
    @wraps(func)
    def yield_wrapper(*args, **kwargs):
        clear_cache()
        yield from func(*args, **kwargs)
        clear_cache()

    @wraps(func)
    def wrapper(*args, **kwargs):
        clear_cache()
        res = func(*args, **kwargs)
        clear_cache()
        return res

    if inspect.isgeneratorfunction(func):
        return yield_wrapper
    else:
        return wrapper


def post_reload_decorator(func):
    @wraps(func)
    def yield_wrapper(*args, **kwargs):
        await_acquire_release_semaphore()
        yield from func(*args, **kwargs)
        async_post_reload_model_weights()

    @wraps(func)
    def wrapper(*args, **kwargs):
        await_acquire_release_semaphore()
        res = func(*args, **kwargs)
        async_post_reload_model_weights()
        return res

    if inspect.isgeneratorfunction(func):
        return yield_wrapper
    else:
        return wrapper


def offload_reload_decorator(func):
    @wraps(func)
    def yield_wrapper(*args, **kwargs):
        await_pre_offload_model_weights()
        yield from func(*args, **kwargs)
        async_post_reload_model_weights()

    @wraps(func)
    def wrapper(*args, **kwargs):
        await_pre_offload_model_weights()
        res = func(*args, **kwargs)
        async_post_reload_model_weights()
        return res

    if inspect.isgeneratorfunction(func):
        return yield_wrapper
    else:
        return wrapper


class torch_default_load_cd(ContextDecorator):
    def __init__(self):
        self.backup_load = safe.load

    def __enter__(self):
        self.backup_load = torch.load
        torch.load = safe.unsafe_torch_load
        return self

    def __exit__(self, *exc):
        torch.load = self.backup_load
        return False
