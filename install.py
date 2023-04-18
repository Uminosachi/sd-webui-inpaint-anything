import launch

if not launch.is_installed("gradio"):
    try:
        launch.run_pip("install gradio", "requirements for gradio")
    except:
        print("Can't install gradio. Please follow the readme to install manually")

if not launch.is_installed("gradio_client"):
    try:
        launch.run_pip("install gradio_client", "requirements for gradio_client")
    except:
        print("Can't install gradio_client. Please follow the readme to install manually")

if not launch.is_installed("accelerate"):
    try:
        launch.run_pip("install accelerate", "requirements for accelerate")
    except:
        print("Can't install accelerate. Please follow the readme to install manually")

if not launch.is_installed("diffusers"):
    try:
        launch.run_pip("install diffusers", "requirements for diffusers")
    except:
        print("Can't install diffusers. Please follow the readme to install manually")

if not launch.is_installed("huggingface-hub"):
    try:
        launch.run_pip("install huggingface-hub", "requirements for huggingface-hub")
    except:
        print("Can't install huggingface-hub. Please follow the readme to install manually")

if not launch.is_installed("matplotlib"):
    try:
        launch.run_pip("install matplotlib", "requirements for matplotlib")
    except:
        print("Can't install matplotlib. Please follow the readme to install manually")

if not launch.is_installed("numpy"):
    try:
        launch.run_pip("install numpy", "requirements for numpy")
    except:
        print("Can't install numpy. Please follow the readme to install manually")

if not launch.is_installed("opencv-python"):
    try:
        launch.run_pip("install opencv-python", "requirements for opencv-python")
    except:
        print("Can't install opencv-python. Please follow the readme to install manually")

if not launch.is_installed("Pillow"):
    try:
        launch.run_pip("install Pillow", "requirements for Pillow")
    except:
        print("Can't install Pillow. Please follow the readme to install manually")

if not launch.is_installed("segment_anything"):
    try:
        launch.run_pip("install -e git+https://github.com/facebookresearch/segment-anything.git#egg=segment_anything", "requirements for segment_anything")
    except:
        print("Can't install segment_anything. Please follow the readme to install manually")

if not launch.is_installed("transformers"):
    try:
        launch.run_pip("install transformers", "requirements for transformers")
    except:
        print("Can't install transformers. Please follow the readme to install manually")

if not launch.is_installed("xformers"):
    try:
        launch.run_pip("install xformers", "requirements for xformers")
    except:
        print("Can't install xformers. Please follow the readme to install manually")