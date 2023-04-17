# Inpaint Anything for Stable Diffusion Web UI

Inpaint Anything performs stable diffusion inpainting on a browser UI using any mask selected from the output of [Segment Anything](https://github.com/facebookresearch/segment-anything).


Using Segment Anything enables users to specify masks by simply pointing to the desired areas, instead of manually filling them in. This can increase the efficiency and accuracy of the mask creation process, leading to potentially higher-quality inpainting results while saving time and effort.

## Installation

To install the software, please follow these steps:

1. Open the "Extensions" tab.
2. Select the "Install from URL" option.
3. Enter the URL of this repository in the "URL for extension's git repository" field.
4. Click the "Install" button.
5. Once installation is complete, restart the Web UI.

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

## License

The source code is licensed under the [Apache 2.0 license](LICENSE).