import logging
import warnings
from typing import Union

import numpy as np
from PIL import Image, ImageDraw

warnings.filterwarnings(action="ignore", category=FutureWarning, module="transformers")

ia_logging = logging.getLogger("Inpaint Anything")
ia_logging.setLevel(logging.INFO)
ia_logging.propagate = False

ia_logging_sh = logging.StreamHandler()
ia_logging_sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
ia_logging_sh.setLevel(logging.INFO)
ia_logging.addHandler(ia_logging_sh)


def draw_text_image(
        input_image: Union[np.ndarray, Image.Image],
        draw_text: str,
        ) -> Image.Image:
    input_image = np.array(input_image) if isinstance(input_image, Image.Image) else input_image
    ret_image = Image.fromarray(np.zeros_like(input_image))
    draw_ret_image = ImageDraw.Draw(ret_image)
    draw_ret_image.text((0, 0), draw_text, fill=(224, 224, 224))

    return ret_image
