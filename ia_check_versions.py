import os
from importlib.util import find_spec

import torch
from modules import paths, sd_samplers_common
from packaging.version import parse

try:
    from functools import cached_property
    from importlib.metadata import version
except Exception:
    from pkg_resources import get_distribution
    def version(module_name): return get_distribution(module_name).version
    cached_property = property


def get_module_version(module_name):
    try:
        module_version = version(module_name)
    except Exception:
        module_version = None
    return module_version


def compare_version(version1, version2):
    if not isinstance(version1, str) or not isinstance(version2, str):
        return None

    if parse(version1) > parse(version2):
        return 1
    elif parse(version1) < parse(version2):
        return -1
    else:
        return 0


def compare_module_version(module_name, version_string):
    module_version = get_module_version(module_name)

    result = compare_version(module_version, version_string)
    return result if result is not None else -2


class IACheckVersions:
    @cached_property
    def diffusers_enable_cpu_offload(self):
        if (find_spec("diffusers") is not None and compare_module_version("diffusers", "0.15.0") >= 0 and
                find_spec("accelerate") is not None and compare_module_version("accelerate", "0.17.0") >= 0):
            return True
        else:
            return False

    @cached_property
    def torch_mps_is_available(self):
        if compare_module_version("torch", "2.0.1") < 0:
            if not getattr(torch, "has_mps", False):
                return False
            try:
                torch.zeros(1).to(torch.device("mps"))
                return True
            except Exception:
                return False
        else:
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()

    @cached_property
    def webui_refiner_is_available(self):
        basedir = os.path.join(paths.script_path, "modules", "processing_scripts")
        refiner_script = os.path.join(basedir, "refiner.py")
        if os.path.isfile(refiner_script) and hasattr(sd_samplers_common, "apply_refiner"):
            return True

        return False


ia_check_versions = IACheckVersions()
