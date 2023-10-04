import os
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download
from modules import shared

from ia_config import get_webui_setting
from ia_logging import ia_logging


class IAFileManager:
    DOWNLOAD_COMPLETE = "Download complete"

    def __init__(self) -> None:
        config_save_folder = get_webui_setting("inpaint_anything_save_folder", "inpaint-anything")
        self.folder_is_webui = False if config_save_folder == "inpaint-anything" else True
        if config_save_folder == "inpaint-anything":
            self._ia_outputs_dir = os.path.join(shared.data_path, "outputs", config_save_folder, datetime.now().strftime("%Y-%m-%d"))
        else:
            webui_save_folder = Path(get_webui_setting("outdir_img2img_samples", os.path.join("outputs", "img2img-images")))
            if webui_save_folder.is_absolute():
                self._ia_outputs_dir = os.path.join(str(webui_save_folder), datetime.now().strftime("%Y-%m-%d"))
            else:
                self._ia_outputs_dir = os.path.join(shared.data_path, str(webui_save_folder), datetime.now().strftime("%Y-%m-%d"))

        self._ia_models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

    def update_ia_outputs_dir(self) -> None:
        """Update inpaint-anything outputs directory.

        Returns:
            None
        """
        config_save_folder = get_webui_setting("inpaint_anything_save_folder", "inpaint-anything")
        self.folder_is_webui = False if config_save_folder == "inpaint-anything" else True
        if config_save_folder == "inpaint-anything":
            self._ia_outputs_dir = os.path.join(shared.data_path, "outputs", config_save_folder, datetime.now().strftime("%Y-%m-%d"))
        else:
            webui_save_folder = Path(get_webui_setting("outdir_img2img_samples", os.path.join("outputs", "img2img-images")))
            if webui_save_folder.is_absolute():
                self._ia_outputs_dir = os.path.join(str(webui_save_folder), datetime.now().strftime("%Y-%m-%d"))
            else:
                self._ia_outputs_dir = os.path.join(shared.data_path, str(webui_save_folder), datetime.now().strftime("%Y-%m-%d"))

    @property
    def outputs_dir(self) -> str:
        """Get inpaint-anything outputs directory.

        Returns:
            str: inpaint-anything outputs directory
        """
        self.update_ia_outputs_dir()
        if not os.path.isdir(self._ia_outputs_dir):
            os.makedirs(self._ia_outputs_dir, exist_ok=True)
        return self._ia_outputs_dir

    @property
    def models_dir(self) -> str:
        """Get inpaint-anything models directory.

        Returns:
            str: inpaint-anything models directory
        """
        if not os.path.isdir(self._ia_models_dir):
            os.makedirs(self._ia_models_dir, exist_ok=True)
        return self._ia_models_dir

    @property
    def savename_prefix(self) -> str:
        """Get inpaint-anything savename prefix.

        Returns:
            str: inpaint-anything savename prefix
        """
        config_save_folder = get_webui_setting("inpaint_anything_save_folder", "inpaint-anything")
        self.folder_is_webui = False if config_save_folder == "inpaint-anything" else True
        basename = "inpainta-" if self.folder_is_webui else ""

        return basename + datetime.now().strftime("%Y%m%d-%H%M%S")


ia_file_manager = IAFileManager()


def download_model_from_hf(hf_model_id, local_files_only=False):
    """Download model from HuggingFace Hub.

    Args:
        sam_model_id (str): HuggingFace model id
        local_files_only (bool, optional): If True, use only local files. Defaults to False.

    Returns:
        str: download status
    """
    if not local_files_only:
        ia_logging.info(f"Downloading {hf_model_id}")
    try:
        snapshot_download(repo_id=hf_model_id, local_files_only=local_files_only)
    except FileNotFoundError:
        return f"{hf_model_id} not found, please download"
    except Exception as e:
        return str(e)

    return IAFileManager.DOWNLOAD_COMPLETE
