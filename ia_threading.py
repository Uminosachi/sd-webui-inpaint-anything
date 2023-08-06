import gc
import threading
from functools import wraps

from modules import devices, shared
from modules.sd_models import (load_model, reload_model_weights,
                               unload_model_weights)

backup_ckpt_info = None
model_access_sem = threading.Semaphore(1)


def clear_cache():
    gc.collect()
    devices.torch_gc()


def webui_reload_model_weights(sd_model=None, info=None):
    try:
        reload_model_weights(sd_model=sd_model, info=info)
    except Exception:
        load_model(checkpoint_info=info)


def pre_unload_model_weights(sem):
    global backup_ckpt_info
    with sem:
        if shared.sd_model is not None:
            backup_ckpt_info = shared.sd_model.sd_checkpoint_info
            unload_model_weights()
            clear_cache()


def await_pre_unload_model_weights():
    global model_access_sem
    thread = threading.Thread(target=pre_unload_model_weights, args=(model_access_sem,))
    thread.start()
    thread.join()


def pre_reload_model_weights(sem):
    global backup_ckpt_info
    with sem:
        if shared.sd_model is None:
            if backup_ckpt_info is not None:
                webui_reload_model_weights(info=backup_ckpt_info)
                backup_ckpt_info = None
            else:
                webui_reload_model_weights()


def await_pre_reload_model_weights():
    global model_access_sem
    thread = threading.Thread(target=pre_reload_model_weights, args=(model_access_sem,))
    thread.start()
    thread.join()


def backup_reload_ckpt_info(sem, info):
    global backup_ckpt_info
    with sem:
        if shared.sd_model is not None:
            if info.title != shared.sd_model.sd_checkpoint_info.title:
                backup_ckpt_info = shared.sd_model.sd_checkpoint_info
                webui_reload_model_weights(sd_model=shared.sd_model, info=info)
        else:
            webui_reload_model_weights(info=info)


def await_backup_reload_ckpt_info(info):
    global model_access_sem
    thread = threading.Thread(target=backup_reload_ckpt_info, args=(model_access_sem, info))
    thread.start()
    thread.join()


def post_reload_model_weights(sem):
    global backup_ckpt_info
    with sem:
        if shared.sd_model is None:
            if backup_ckpt_info is not None:
                webui_reload_model_weights(info=backup_ckpt_info)
                backup_ckpt_info = None
            else:
                webui_reload_model_weights()

        elif backup_ckpt_info is not None:
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
    def wrapper(*args, **kwargs):
        clear_cache()
        res = func(*args, **kwargs)
        clear_cache()
        return res

    return wrapper


def post_reload_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        await_acquire_release_semaphore()
        res = func(*args, **kwargs)
        async_post_reload_model_weights()
        return res

    return wrapper
