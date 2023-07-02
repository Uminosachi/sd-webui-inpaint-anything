import configparser
import os
from ia_ui_items import get_sam_model_ids, get_model_ids

ia_config_ini_path = os.path.join(os.path.dirname(__file__), "ia_config.ini")
class IAConfig:
    KEYS = ["sam_model_id", "inp_model_id"]
    
    KEY_SAM_MODEL_ID = "sam_model_id"
    KEY_INP_MODEL_ID = "inp_model_id"
    
    SECTION_DEFAULT = "DEFAULT"
    SECTION_USER = "USER"

def setup_ia_config_ini():
    global ia_config_ini_path
    if not os.path.isfile(ia_config_ini_path):
        ia_config_ini = configparser.ConfigParser()
        
        sam_model_ids = get_sam_model_ids()
        sam_model_index = sam_model_ids.index("sam_vit_l_0b3195.pth") if "sam_vit_l_0b3195.pth" in sam_model_ids else 1
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
    
    return None

def set_ia_config(key, option, section=None):
    global ia_config_ini_path
    if not os.path.isfile(ia_config_ini_path):
        setup_ia_config_ini()
    
    if section is None:
        section = IAConfig.SECTION_DEFAULT
    
    ia_config_ini = configparser.ConfigParser()
    ia_config_ini.read(ia_config_ini_path)
    
    if not ia_config_ini.has_section(section):
        ia_config_ini[section] = {}
    
    ia_config_ini[section][key] = option
    
    with open(ia_config_ini_path, "w") as f:
        ia_config_ini.write(f)
