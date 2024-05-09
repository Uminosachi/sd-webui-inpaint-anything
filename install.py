import launch

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
}

for package in required_packages:
    if not launch.is_installed(package):
        try:
            launch.run_pip(f"install {required_packages[package]}", f"requirements for {package}")
        except Exception:
            print(f"Can't install {required_packages[package]}. Please follow the readme to install manually")
