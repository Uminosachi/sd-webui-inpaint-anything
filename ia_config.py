import configparser
import json
import os

from modules import shared

from ia_ui_items import get_inp_model_ids, get_sam_model_ids


class IAConfig:
    SECTION_DEFAULT = "DEFAULT"
    SECTION_USER = "USER"

    SECTIONS = [SECTION_DEFAULT, SECTION_USER]

    KEY_SAM_MODEL_ID = "sam_model_id"
    KEY_INP_MODEL_ID = "inp_model_id"

    KEYS = [KEY_SAM_MODEL_ID, KEY_INP_MODEL_ID]

    KEY_WEBUI_SAM_MODEL_ID = "inpaint_anything/Segment Anything Model ID/value"
    KEY_WEBUI_INP_MODEL_ID = "inpaint_anything/Inpainting Model ID/value"

    INI_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ia_config.ini")
    WEBUI_CONFIG_PATH = os.path.join(shared.data_path, "ui-config.json")


def setup_ia_config_ini():
    if not os.path.isfile(IAConfig.INI_PATH):
        ia_config_ini = configparser.ConfigParser()

        sam_model_ids = get_sam_model_ids()
        sam_model_index = 1
        inp_model_ids = get_inp_model_ids()
        inp_model_index = 0

        ia_config_ini[IAConfig.SECTION_DEFAULT] = {
            IAConfig.KEY_SAM_MODEL_ID: sam_model_ids[sam_model_index],
            IAConfig.KEY_INP_MODEL_ID: inp_model_ids[inp_model_index],
        }
        with open(IAConfig.INI_PATH, "w", encoding="utf-8") as f:
            ia_config_ini.write(f)


def get_ia_config(key, section=IAConfig.SECTION_DEFAULT):
    setup_ia_config_ini()

    ia_config_ini = configparser.ConfigParser()
    ia_config_ini.read(IAConfig.INI_PATH, encoding="utf-8")

    if ia_config_ini.has_option(section, key):
        return ia_config_ini[section][key]

    section = IAConfig.SECTION_DEFAULT
    if ia_config_ini.has_option(section, key):
        return ia_config_ini[section][key]

    return None


def get_ia_config_index(key, section=IAConfig.SECTION_DEFAULT):
    value = get_ia_config(key, section)

    if value is None:
        return None

    if key == IAConfig.KEY_SAM_MODEL_ID:
        sam_model_ids = get_sam_model_ids()
        idx = sam_model_ids.index(value) if value in sam_model_ids else 1
    elif key == IAConfig.KEY_INP_MODEL_ID:
        inp_model_ids = get_inp_model_ids()
        idx = inp_model_ids.index(value) if value in inp_model_ids else 0
    else:
        idx = None

    return idx


def set_ia_config(key, value, section=IAConfig.SECTION_DEFAULT):
    setup_ia_config_ini()

    ia_config_ini = configparser.ConfigParser()
    ia_config_ini.read(IAConfig.INI_PATH, encoding="utf-8")

    if section != IAConfig.SECTION_DEFAULT and not ia_config_ini.has_section(section):
        ia_config_ini[section] = {}
    else:
        if ia_config_ini.has_option(section, key) and ia_config_ini[section][key] == value:
            return

    try:
        ia_config_ini[section][key] = value
    except Exception:
        ia_config_ini[section] = {}
        ia_config_ini[section][key] = value

    with open(IAConfig.INI_PATH, "w", encoding="utf-8") as f:
        ia_config_ini.write(f)

    if os.path.isfile(IAConfig.WEBUI_CONFIG_PATH):
        with open(IAConfig.WEBUI_CONFIG_PATH, "r", encoding="utf-8") as f:
            webui_config = json.load(f)

        if key == IAConfig.KEY_SAM_MODEL_ID:
            if IAConfig.KEY_WEBUI_SAM_MODEL_ID in webui_config.keys():
                webui_config[IAConfig.KEY_WEBUI_SAM_MODEL_ID] = value
        elif key == IAConfig.KEY_INP_MODEL_ID:
            if IAConfig.KEY_WEBUI_INP_MODEL_ID in webui_config.keys():
                webui_config[IAConfig.KEY_WEBUI_INP_MODEL_ID] = value

        with open(IAConfig.WEBUI_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(webui_config, f, indent=4)
