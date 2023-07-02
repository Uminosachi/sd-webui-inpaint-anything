import configparser
import os
from ia_ui_items import get_sam_model_ids, get_model_ids
from modules.paths_internal import data_path
import json

ia_config_ini_path = os.path.join(os.path.dirname(__file__), "ia_config.ini")
webui_config_path = os.path.join(data_path, "ui-config.json")

class IAConfig:
    SECTIONS = ["DEFAULT", "USER"]
    KEYS = ["sam_model_id", "inp_model_id"]

    SECTION_DEFAULT = "DEFAULT"
    SECTION_USER = "USER"

    KEY_SAM_MODEL_ID = "sam_model_id"
    KEY_INP_MODEL_ID = "inp_model_id"
    
    KEY_WEBUI_SAM_MODEL_ID = "inpaint_anything/Segment Anything Model ID/value"
    KEY_WEBUI_INP_MODEL_ID = "inpaint_anything/Inpainting Model ID/value"

def setup_ia_config_ini():
    global ia_config_ini_path
    if not os.path.isfile(ia_config_ini_path):
        ia_config_ini = configparser.ConfigParser()
        
        sam_model_ids = get_sam_model_ids()
        sam_model_index = 1
        inp_model_ids = get_model_ids()
        inp_model_index = 0
        
        ia_config_ini[IAConfig.SECTION_DEFAULT] = {
            IAConfig.KEY_SAM_MODEL_ID: sam_model_ids[sam_model_index],
            IAConfig.KEY_INP_MODEL_ID: inp_model_ids[inp_model_index],
        }
        with open(ia_config_ini_path, "w") as f:
            ia_config_ini.write(f)

def get_ia_config(key, section=None):
    global ia_config_ini_path
    if not os.path.isfile(ia_config_ini_path):
        setup_ia_config_ini()
    
    if section is None:
        section = IAConfig.SECTION_DEFAULT
    
    ia_config_ini = configparser.ConfigParser()
    ia_config_ini.read(ia_config_ini_path)
    
    if ia_config_ini.has_section(section) and ia_config_ini.has_option(section, key):
        return ia_config_ini[section][key]

    section = IAConfig.SECTION_DEFAULT
    if ia_config_ini.has_option(section, key):
        return ia_config_ini[section][key]
    
    return None

def get_ia_config_index(key, section=None):
    option = get_ia_config(key, section)
    
    if option is None:
        return None
    
    if key == IAConfig.KEY_SAM_MODEL_ID:
        sam_model_ids = get_sam_model_ids()
        idx = sam_model_ids.index(option) if option in sam_model_ids else 1
        # print("ia_config: get_ia_config_index: key: {}, option: {}, idx: {}".format(key, option, idx))
    elif key == IAConfig.KEY_INP_MODEL_ID:
        inp_model_ids = get_model_ids()
        idx = inp_model_ids.index(option) if option in inp_model_ids else 0
        # print("ia_config: get_ia_config_index: key: {}, option: {}, idx: {}".format(key, option, idx))
    else:
        idx = None

    return idx

def set_ia_config(key, option, section=None):
    global ia_config_ini_path
    global webui_config_path
    if not os.path.isfile(ia_config_ini_path):
        setup_ia_config_ini()
    
    if section is None:
        section = IAConfig.SECTION_DEFAULT
    
    ia_config_ini = configparser.ConfigParser()
    ia_config_ini.read(ia_config_ini_path)
    
    if section != IAConfig.SECTION_DEFAULT and not ia_config_ini.has_section(section):
        ia_config_ini[section] = {}
    else:
        if ia_config_ini.has_option(section, key) and ia_config_ini[section][key] == option:
            return
    
    # print("ia_config: set_ia_config: section: {}, key: {}, option: {}".format(section, key, option))
    ia_config_ini[section][key] = option
    
    with open(ia_config_ini_path, "w") as f:
        ia_config_ini.write(f)
    
    if os.path.isfile(webui_config_path):
        with open(webui_config_path, "r", encoding="utf-8") as f:
            webui_config = json.load(f)
        
        if key == IAConfig.KEY_SAM_MODEL_ID:
            if IAConfig.KEY_WEBUI_SAM_MODEL_ID in webui_config.keys():
                webui_config[IAConfig.KEY_WEBUI_SAM_MODEL_ID] = option
        elif key == IAConfig.KEY_INP_MODEL_ID:
            if IAConfig.KEY_WEBUI_INP_MODEL_ID in webui_config.keys():
                webui_config[IAConfig.KEY_WEBUI_INP_MODEL_ID] = option
        
        with open(webui_config_path, "w", encoding="utf-8") as f:
            json.dump(webui_config, f, indent=4)
