# Tooling to run ComfyUI pipelines as a Replicate Cog model

This is an implementation of the ComfyUI workflow as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

The workflows used for this repo are found under:

    custom_workflows/*.json

## Usage:
1. Manually run ComfyUI and install all the custom nodes you need for your pipeline (this will clone the dependencies under ComfyUI/custom_nodes
2. Download any required checkpoints and place them in the correct folders
3. If you're adding a new pipeline, you need to:
    - create two new files under 'custom_workflows': yourname_api.json and yourname_inputs.json
    - the first file is simply your comfyUI workflow, exported as API format (make sure to delete all the preview nodes and make sure there is one single output node)
    - the second file maps the ComfyUI nodes to the cog inputs
    - update predict.py to accomodate for your new pipeline
  
## Examples
sudo cog predict -i mode="comfy_makeitrad" -i text_input="a swimming pool in the style of embedding:makeitrad_embeddings" -i width="512"


sudo cog predict -i mode="comfy_img2vid" -i init_image="https://storage.googleapis.com/public-assets-xander/A_workbox/47c1a72ef146d5fe6d4ccd9dcaa16fca7b77723e3b6cb1668ccfa92dba7ac86a%20(2).jpg" -i width="512" -i height="512" -i n_frames="16"

sudo cog predict -i mode="comfy_txt2vid" -i interpolation_texts="a big elephant|a super fast racecar" -i n_frames="32" -i width="512" -i height="512"

sudo cog predict -i mode="comfy_upscale" -i init_image="https://storage.googleapis.com/public-assets-xander/A_workbox/47c1a72ef146d5fe6d4ccd9dcaa16fca7b77723e3b6cb1668ccfa92dba7ac86a%20(2).jpg" -i width="1600" -i height="1600" -i denoise_strength="0.6"

sudo cog predict -i mode="comfy_vid2vid" -i init_image="https://storage.googleapis.com/public-assets-xander/A_workbox/47c1a72ef146d5fe6d4ccd9dcaa16fca7b77723e3b6cb1668ccfa92dba7ac86a%20(2).jpg" -i input_video_path="https://storage.googleapis.com/public-assets-xander/A_workbox/flowerspiral.gif" -i width="1024" -i height="1024" -i denoise_strength="0.9"

![alt text](output.png)
