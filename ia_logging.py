import logging
import warnings

warnings.filterwarnings(action="ignore", category=FutureWarning, module="transformers")

ia_logging = logging.getLogger("Inpaint Anything")
ia_logging.setLevel(logging.INFO)
ia_logging.propagate = False

ia_logging_sh = logging.StreamHandler()
ia_logging_sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
ia_logging_sh.setLevel(logging.INFO)
ia_logging.addHandler(ia_logging_sh)
