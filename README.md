# Inpaint Anything for Stable Diffusion Web UI

Inpaint Anything performs stable diffusion inpainting on a browser UI using any mask selected from the output of [Segment Anything](https://github.com/facebookresearch/segment-anything).


Using Segment Anything enables users to specify masks by simply pointing to the desired areas, instead of manually filling them in. This can increase the efficiency and accuracy of the mask creation process, leading to potentially higher-quality inpainting results while saving time and effort.

[Standalone version](https://github.com/Uminosachi/inpaint-anything)

## Installation

To install the software, please follow these steps:

1. Open the "Extensions" tab on the AUTOMATIC1111's [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui.git).
2. Select the "Install from URL" option.
3. Enter `https://github.com/Uminosachi/sd-webui-inpaint-anything.git` in the "URL for extension's git repository" field.
4. Click the "Install" button.
5. Once installation is complete, restart the Web UI.

## Running the application

* To use xFormers for inference, please add the `--xformers` argument to the startup command. For example, run `./webui.sh --xformers` or `webui.bat --xformers`
* Note: If you have a privacy protection extension enabled in your web browser, such as DuckDuckGo, you may not be able to retrieve the mask from your sketch.
* Note: In Gradio version 3.23.0 or older, the segmentation image may appear small on the Web UI.

## Downloading the Model

To download the model:

1. Go to the "Inpaint Anything" tab of the Web UI.
2. Click on the "Download model" button next to the [Segment Anything Model ID](https://github.com/facebookresearch/segment-anything#model-checkpoints).
3. Wait for the download to complete.
4. The downloaded model file will be stored in the `models` directory of this application's repository.

## Usage

* Drag and drop your image onto the input image area.
* Click the "Run Segment Anything" button.
* Use sketching to define the area you want to inpaint. You can undo and adjust the pen size.
* Click the "Create mask" button (the mask will appear in the selected mask image area).
* Enter the Prompt and Negative Prompt, Choose the Inpainting Model ID.
* Click the "Run Inpainting" button (**Please note that it may take some time to download the model for the first time**).
* You can change the Sampling Steps, the Guidance Scale and the Seed in the Advanced options.

Inpainting is performed using [diffusers](https://github.com/huggingface/diffusers).

![UI image](images/inpaint_anything_ui_image_1.png)

## Auto-saving images

* The inpainted image will be automatically saved in the folder that matches the current date within the `outputs/inpaint-anything` directory.

## License

The source code is licensed under the [Apache 2.0 license](LICENSE).
