# Tooling to run ComfyUI pipelines as a Replicate Cog model

This is an implementation of the ComfyUI workflow as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

The workflows used for this repo are found under:

    custom_workflows/*.json

## Usage:
1. `git clone https://github.com/comfyanonymous/ComfyUI.git` in the root directory of this repo
2. Manually run your ComfyUI pipeline to verify everything works (`python main.py`), you can install all the custom nodes you need for your pipeline (this will clone the dependencies under ComfyUI/custom_nodes
3. Download any required checkpoints and place them in the correct folders
4. If you're adding a new pipeline, you need to:
    - create two new files under 'custom_workflows': yourname_api.json and yourname_inputs.json
    - the first file is simply your comfyUI workflow, exported as API format (make sure to delete all the preview nodes and make sure there is one single output node)
    - the second file maps the ComfyUI nodes to the cog inputs
    - update predict.py to accomodate for your new pipeline
5. (after installing cog) build the cog image: ``cog build``
  
## Examples

`sudo cog predict -i mode="comfy_img2vid" -i init_image="YOUR_PUBLIC_IMG_URL" -i n_frames="32"`

``sudo cog predict -i mode="comfy_txt2vid" -i interpolation_texts="a big elephant|a super fast racecar" -i n_frames="32"``

``sudo cog predict -i mode="comfy_upscale" -i init_image="YOUR_PUBLIC_IMG_URL"``

``sudo cog predict -i mode="comfy_vid2vid" -i init_image="YOUR_PUBLIC_IMG_URL" -i input_video_path="YOUR_PUBLIC_GIF_URL"``
