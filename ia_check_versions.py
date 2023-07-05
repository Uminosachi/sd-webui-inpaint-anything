from packaging.version import parse
from importlib.util import find_spec
try:
    from importlib.metadata import version
    from functools import cached_property
except:
    from pkg_resources import get_distribution
    version = lambda module_name: get_distribution(module_name).version
    cached_property = property

def get_module_version(module_name):
    module_version = version(module_name) if find_spec(module_name) is not None else None
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

ia_check_versions = IACheckVersions()
