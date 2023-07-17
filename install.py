import launch

if not launch.is_installed("accelerate"):
    try:
        launch.run_pip("install accelerate", "requirements for accelerate")
    except Exception:
        print("Can't install accelerate. Please follow the readme to install manually")

if not launch.is_installed("diffusers"):
    try:
        launch.run_pip("install diffusers", "requirements for diffusers")
    except Exception:
        print("Can't install diffusers. Please follow the readme to install manually")

if not launch.is_installed("huggingface_hub"):
    try:
        launch.run_pip("install huggingface-hub", "requirements for huggingface_hub")
    except Exception:
        print("Can't install huggingface-hub. Please follow the readme to install manually")

if not launch.is_installed("numpy"):
    try:
        launch.run_pip("install numpy", "requirements for numpy")
    except Exception:
        print("Can't install numpy. Please follow the readme to install manually")

if not launch.is_installed("cv2"):
    try:
        launch.run_pip("install opencv-python", "requirements for cv2")
    except Exception:
        print("Can't install opencv-python. Please follow the readme to install manually")

if not launch.is_installed("PIL"):
    try:
        launch.run_pip("install Pillow", "requirements for PIL")
    except Exception:
        print("Can't install Pillow. Please follow the readme to install manually")

if not launch.is_installed("segment_anything"):
    try:
        launch.run_pip("install segment-anything", "requirements for segment_anything")
    except Exception:
        print("Can't install segment-anything. Please follow the readme to install manually")

if not launch.is_installed("transformers"):
    try:
        launch.run_pip("install transformers", "requirements for transformers")
    except Exception:
        print("Can't install transformers. Please follow the readme to install manually")

if not launch.is_installed("lama_cleaner"):
    try:
        launch.run_pip("install lama-cleaner", "requirements for lama_cleaner")
    except Exception:
        print("Can't install lama-cleaner. Please follow the readme to install manually")

if not launch.is_installed("ultralytics"):
    try:
        launch.run_pip("install ultralytics", "requirements for ultralytics")
    except Exception:
        print("Can't install ultralytics. Please follow the readme to install manually")

if not launch.is_installed("tqdm"):
    try:
        launch.run_pip("install tqdm", "requirements for tqdm")
    except Exception:
        print("Can't install tqdm. Please follow the readme to install manually")

if not launch.is_installed("packaging"):
    try:
        launch.run_pip("install packaging", "requirements for packaging")
    except Exception:
        print("Can't install packaging. Please follow the readme to install manually")
