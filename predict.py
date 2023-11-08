# don't push DEBUG_MODE = True to Replicate!
DEBUG_MODE = False


import subprocess
import threading
import time
from cog import BasePredictor, BaseModel, Path, Input
from typing import Iterator, Optional, List
import os
import torch
import shutil
import uuid
import json
import urllib
import websocket
from PIL import Image
from urllib.error import URLError
import random
from PIL import Image
import io

class CogOutput(BaseModel):
    files: List[Path]
    name: Optional[str] = None
    thumbnails: Optional[List[Path]] = None
    attributes: Optional[dict] = None
    progress: Optional[float] = None
    isFinal: bool = False

class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

class Predictor(BasePredictor):

    GENERATOR_OUTPUT_TYPE = Path if DEBUG_MODE else CogOutput

    def setup(self):
        # start server
        self.server_address = "127.0.0.1:8188"
        self.start_server()

    def start_server(self):
        server_thread = threading.Thread(target=self.run_server)
        server_thread.start()

        while not self.is_server_running():
            time.sleep(1)  # Wait for 1 second before checking again

        print("Server is up and running!")

    def run_server(self):
        command = "python ./ComfyUI/main.py"
        server_process = subprocess.Popen(command, shell=True)
        server_process.wait()

    # hacky solution, will fix later
    def is_server_running(self):
        try:
            with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, "123")) as response:
                return response.status == 200
        except URLError:
            return False
    
    def queue_prompt(self, prompt, client_id):
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req =  urllib.request.Request("http://{}/prompt".format(self.server_address), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        print(folder_type)
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(self.server_address, url_values)) as response:
            return response.read()

    def get_video(self, filename, subfolder):
        data = {'gifs': [{'filename': filename, 'subfolder': subfolder, 'type': 'output', 'format': 'video/h264-mp4'}]}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(self.server_address, url_values)) as response:
            return response.read()

    def get_output(self, ws, prompt, client_id):
        prompt_id = self.queue_prompt(prompt, client_id)['prompt_id']
        output_paths = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break #Execution is done
            else:
                continue #previews are binary data
        
        # hardcoded for now: TODO make this more flexible
        output_dir = "ComfyUI/output"

        history = self.get_history(prompt_id)[prompt_id]
        for o in history['outputs']:
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                print("node output: ", node_output)

                if 'images' in node_output:
                    outputs = []
                    for image in node_output['images']:
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        outputs.append(image_data)
                
                if 'gifs' in node_output:
                    outputs = []
                    for video in node_output['gifs']:
                        filename, subfolder = video['filename'], video['subfolder']
                        # get full video path:
                        video_path = os.path.join(output_dir, subfolder, filename)
                        outputs.append(video_path)
                        
                output_paths[node_id] = outputs

        return output_paths

    def get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, prompt_id)) as response:
            return json.loads(response.read())

    def get_workflow_output(self, args):
        
        if args.mode == "vid2vid":
            workflow_config = "./custom_workflows/eden_vid2vid_api.json"

        # load config
        with open(workflow_config, 'r') as file:
            prompt = json.load(file)

        # video input:
        prompt["55"]["inputs"]["video"] = args.input_video_path
        prompt["55"]["inputs"]["frame_load_cap"]  = args.n_frames

        prompt["3"]["inputs"]["text"] = args.prompt
        prompt["6"]["inputs"]["text"] = args.negative_prompt

        # sampler:
        prompt["7"]["inputs"]["steps"] = args.steps
        prompt["7"]["inputs"]["cfg"]   = args.guidance_scale
        prompt["7"]["inputs"]["seed"]  = args.seed

        prompt["9"]["inputs"]["width"]  = args.width
        prompt["9"]["inputs"]["height"] = args.height

        # controlnet:
        prompt["20"]["inputs"]["control_net_name"] = args.controlnet_type
        prompt["24"]["inputs"]["strength"] = args.controlnet_strength

        # pretty print final config:
        print("------------------------------------------")
        print("------------------------------------------")
        print(json.dumps(prompt, indent=4, sort_keys=True))

        # start the process
        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(self.server_address, client_id))
        out_paths = self.get_output(ws, prompt, client_id)

        for node_id in out_paths:
            for out_path in out_paths[node_id]:
                return Path(out_path)
    
    def predict(
        self,
        input_video_path: str = Input(
                    description="Load source video from file, url, or base64 string", 
                ),
        prompt: str = Input(description="Prompt", default="the tree of life"),
        negative_prompt: str = Input(description="Negative Prompt", default="nude, naked, text, watermark, low-quality, signature, padding, margins, white borders, padded border, moiré pattern, downsampling, aliasing, distorted, blurry, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, grainy, error, bad-contrast"),
        steps: int = Input(
            description="Steps",
            default=20
        ),
        width: int = Input(
            description="Width", 
            ge=512, le=2048, default=512
        ),
        height: int = Input(
            description="Height", 
            ge=512, le=2048, default=512
        ),

        n_frames: int = Input(
            description="Total number of frames (mode==interpolate)",
            ge=16, le=64, default=16
        ),

        guidance_scale: float = Input(
            description="Strength of text conditioning guidance", 
            ge=1, le=20, default=7.5
        ),

        mode: str = Input(
            description="Mode", default="vid2vid",
            choices=["vid2vid"]
        ),

        controlnet_type: str = Input(
            description="Controlnet type",
            default="canny-edge",
            choices=["canny-edge", "qr_monster"]
        ),
        controlnet_strength: float = Input(
            description="Strength of controlnet guidance", 
            ge=0.0, le=1.5, default=0.8
        ),

        seed: int = Input(description="Sampling seed, leave Empty for Random", default=None),

    ) -> Iterator[GENERATOR_OUTPUT_TYPE]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(3), "big")
        generator = torch.Generator("cuda").manual_seed(seed)

        controlnet_map = {
            "canny-edge": "control_sd15_canny.safetensors",
            "qr_monster": "control_v1p_sd15_qrcode_monster.safetensors",
        }

        # gather args from the input fields:
        args = {
            "input_video_path": input_video_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "n_frames": n_frames,
            "guidance_scale": guidance_scale,
            "mode": mode,
            "controlnet_type": controlnet_map[controlnet_type],
            "controlnet_strength": controlnet_strength,
            "steps": steps,
            "seed": seed,
        }

        # Example usage:
        args = AttrDict(args)

        # queue prompt
        output_path = self.get_workflow_output(args)
        print("----------------------------------------------")
        print("Received final output:")
        print(output_path)

        if DEBUG_MODE:
            yield Path(output_path)
        else:
            yield CogOutput(files=[Path(output_path)], name=prompt, thumbnails=[Path(output_path)], attributes=None, progress=1.0, isFinal=True)
