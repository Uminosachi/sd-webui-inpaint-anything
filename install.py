import subprocess
import sys

required_packages = {
    "accelerate": "accelerate",
    "diffusers": "diffusers",
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
}

for package, install_name in required_packages.items():
    try:
        __import__(package)
        # print(f"{package} is already installed.")
    except ImportError:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", install_name], check=True)
            print(f"Successfully installed {install_name}.")
        except subprocess.CalledProcessError:
            print(f"Can't install {install_name}. Please follow the readme to install manually.")
