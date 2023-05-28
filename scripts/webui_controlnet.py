import importlib

def find_controlnet():
    try:
        cnet = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
    except:
        try:
            cnet = importlib.import_module('extensions-builtin.sd-webui-controlnet.scripts.external_code', 'external_code')
        except:
            cnet = None
    
    return cnet
