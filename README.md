# Tooling to run ComfyUI pipelines as a Replicate Cog model

Any ComfyUI pipeline integrated here can easily be run on [Eden](https://eden.art/)

This is an implementation of the ComfyUI workflow as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

The workflows used for this repo are found under:

    custom_workflows/*.json

## Usage:
1. `git clone https://github.com/comfyanonymous/ComfyUI.git` in the root directory of this repo
2. Manually run your ComfyUI pipeline to verify everything works (`python main.py`), you can install all the custom nodes you need for your pipeline (this will clone the dependencies under ComfyUI/custom_nodes
3. Download any required checkpoints and place them in the correct folders
4. If you're adding a new pipeline, you need to:
    - create two new files under 'custom_workflows': pipelinename_api.json and pipelinename_inputs.json
    - the first file is simply your comfyUI workflow, [exported as API format](https://github.com/comfyanonymous/ComfyUI/blob/master/script_examples/basic_api_example.py#L7C65-L7C65) (make sure to delete all the preview nodes in your pipeline and make sure there is one single output node that saves to disk)
    - the second file maps the ComfyUI input nodes to the cog inputs, you have to manually fill this in, it's a bit tedious, should be automate-able at some point
    - update predict.py to accomodate for your new pipeline
5. (after installing cog) build the cog image: ``cog build`` and test your pipeline as the examples below show:
  
## Examples

`sudo cog predict -i mode="comfy_img2vid" -i init_image="YOUR_PUBLIC_IMG_URL" -i n_frames="32"`

``sudo cog predict -i mode="comfy_txt2vid" -i interpolation_texts="a big elephant|a super fast racecar" -i n_frames="32"``

``sudo cog predict -i mode="comfy_upscale" -i init_image="YOUR_PUBLIC_IMG_URL"``

``sudo cog predict -i mode="comfy_vid2vid" -i init_image="YOUR_PUBLIC_IMG_URL" -i input_video_path="YOUR_PUBLIC_GIF_URL"``
