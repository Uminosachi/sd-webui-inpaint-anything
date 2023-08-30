import configparser
import json
import os
from types import SimpleNamespace

from modules import shared

from ia_ui_items import get_inp_model_ids, get_inp_webui_model_ids, get_sam_model_ids


class IAConfig:
    SECTIONS = SimpleNamespace(
        DEFAULT=configparser.DEFAULTSECT,
        USER="USER",
    )

    KEYS = SimpleNamespace(
        SAM_MODEL_ID="sam_model_id",
        INP_MODEL_ID="inp_model_id",
        INP_WEBUI_MODEL_ID="inp_webui_model_id",
    )

    PATHS = SimpleNamespace(
        INI=os.path.join(os.path.dirname(os.path.realpath(__file__)), "ia_config.ini"),
        WEBUI_CONFIG=os.path.join(shared.data_path, "ui-config.json"),
    )

    def __init__(self):
        self.ids_dict = {}
        self.ids_dict[IAConfig.KEYS.SAM_MODEL_ID] = {
            "list": get_sam_model_ids(),
            "index": 1,
        }
        self.ids_dict[IAConfig.KEYS.INP_MODEL_ID] = {
            "list": get_inp_model_ids(),
            "index": 0,
        }
        self.ids_dict[IAConfig.KEYS.INP_WEBUI_MODEL_ID] = {
            "list": get_inp_webui_model_ids(),
            "index": 0,
        }

        self.webui_keys = {}
        self.webui_keys[IAConfig.KEYS.SAM_MODEL_ID] = "inpaint_anything/Segment Anything Model ID/value"
        self.webui_keys[IAConfig.KEYS.INP_MODEL_ID] = "inpaint_anything/Inpainting Model ID/value"
        self.webui_keys[IAConfig.KEYS.INP_WEBUI_MODEL_ID] = "inpaint_anything/Inpainting Model ID webui/value"


ia_config = IAConfig()


def setup_ia_config_ini():
    ia_config_ini = configparser.ConfigParser(defaults={})
    if os.path.isfile(IAConfig.PATHS.INI):
        ia_config_ini.read(IAConfig.PATHS.INI, encoding="utf-8")

    changed = False
    for key, ids_info in ia_config.ids_dict.items():
        if not ia_config_ini.has_option(IAConfig.SECTIONS.DEFAULT, key):
            if len(ids_info["list"]) > ids_info["index"]:
                ia_config_ini[IAConfig.SECTIONS.DEFAULT][key] = ids_info["list"][ids_info["index"]]
                changed = True
        else:
            if len(ids_info["list"]) > ids_info["index"] and ia_config_ini[IAConfig.SECTIONS.DEFAULT][key] != ids_info["list"][ids_info["index"]]:
                ia_config_ini[IAConfig.SECTIONS.DEFAULT][key] = ids_info["list"][ids_info["index"]]
                changed = True

    if changed:
        with open(IAConfig.PATHS.INI, "w", encoding="utf-8") as f:
            ia_config_ini.write(f)


def get_ia_config(key, section=IAConfig.SECTIONS.DEFAULT):
    setup_ia_config_ini()

    ia_config_ini = configparser.ConfigParser(defaults={})
    ia_config_ini.read(IAConfig.PATHS.INI, encoding="utf-8")

    if ia_config_ini.has_option(section, key):
        return ia_config_ini[section][key]

    section = IAConfig.SECTIONS.DEFAULT
    if ia_config_ini.has_option(section, key):
        return ia_config_ini[section][key]

    return None


def get_ia_config_index(key, section=IAConfig.SECTIONS.DEFAULT):
    value = get_ia_config(key, section)

    ids_dict = ia_config.ids_dict
    if value is None:
        if key in ids_dict.keys():
            ids_info = ids_dict[key]
            return ids_info["index"]
        else:
            return 0
    else:
        if key in ids_dict.keys():
            ids_info = ids_dict[key]
            return ids_info["list"].index(value) if value in ids_info["list"] else ids_info["index"]
        else:
            return 0


def set_ia_config(key, value, section=IAConfig.SECTIONS.DEFAULT):
    setup_ia_config_ini()

    ia_config_ini = configparser.ConfigParser(defaults={})
    ia_config_ini.read(IAConfig.PATHS.INI, encoding="utf-8")

    if ia_config_ini.has_option(section, key) and ia_config_ini[section][key] == value:
        return

    if section != IAConfig.SECTIONS.DEFAULT and not ia_config_ini.has_section(section):
        ia_config_ini[section] = {}

    try:
        ia_config_ini[section][key] = value
    except Exception:
        ia_config_ini[section] = {}
        ia_config_ini[section][key] = value

    with open(IAConfig.PATHS.INI, "w", encoding="utf-8") as f:
        ia_config_ini.write(f)

    if os.path.isfile(IAConfig.PATHS.WEBUI_CONFIG):
        try:
            with open(IAConfig.PATHS.WEBUI_CONFIG, "r", encoding="utf-8") as f:
                webui_config = json.load(f)

            webui_keys = ia_config.webui_keys
            if key in webui_keys.keys() and webui_keys[key] in webui_config.keys():
                webui_config[webui_keys[key]] = value

                with open(IAConfig.PATHS.WEBUI_CONFIG, "w", encoding="utf-8") as f:
                    json.dump(webui_config, f, indent=4)

        except Exception:
            pass


def get_webui_setting(key, default):
    value = shared.opts.data.get(key, default)

    if not isinstance(value, type(default)):
        value = default

    return value
