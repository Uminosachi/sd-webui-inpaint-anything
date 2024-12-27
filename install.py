import subprocess
import sys

required_packages = {
    "accelerate": "accelerate",
    "diffusers": "diffusers<0.32.0",
    "huggingface_hub": "huggingface-hub",
    "numpy": "numpy",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "segment_anything": "segment-anything",
    "transformers": "transformers",
    "ultralytics": "ultralytics",
    "tqdm": "tqdm",
    "packaging": "packaging",
    "loguru": "loguru",
    "rich": "rich",
    "pydantic": "pydantic",
    "timm": "timm",
    "onnxruntime": "onnxruntime",
    "hydra": "hydra-core",
    "iopath": "iopath",
}

for package, install_name in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", install_name], check=True)
            print(f"Successfully installed {install_name}.")
        except subprocess.CalledProcessError:
            print(f"Can't install {install_name}. Please follow the readme to install manually.")
