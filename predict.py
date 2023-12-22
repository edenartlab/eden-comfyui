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
import requests
from PIL import Image
from urllib.error import URLError
import random
from PIL import Image
import io

import cv2
import tempfile
import subprocess
import os
import shlex

def save_first_frame_to_tempfile(video_path):
    if not video_path.endswith('.mp4'):
        # the path is not a video file, just return it
        return video_path

    # Capture the video from the file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error reading the first frame")
        return

    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    cv2.imwrite(temp_file.name, frame)
    cap.release()

    return temp_file.name

def reencode_video(input_file_path, output_file_path):
    input_file_path_str = str(input_file_path)  # Convert Path object to string

    # Input validation
    if not os.path.exists(input_file_path_str) or os.path.getsize(input_file_path_str) == 0:
        print("Invalid input file.")
        return

    # Determine if the input is a GIF
    is_gif = input_file_path_str.lower().endswith('.gif')

    # Base FFmpeg command with scale filter, using triple quotes for clarity
    base_command = f"""
        ffmpeg -y -err_detect ignore_err -i "{input_file_path}" -vf "scale='min(2048,iw)':'min(2048,ih)':force_original_aspect_ratio=decrease"
        -c:v libx264 -preset medium
    """

    # Exclude audio codec for GIF files
    if not is_gif:
        base_command += ' -c:a aac'

    base_command += f' "{output_file_path}"'

    try:
        # Run the command with a timeout (e.g., 300 seconds)
        subprocess.run(shlex.split(base_command), check=True, timeout=300, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Re-encoding completed: {output_file_path}")
    except subprocess.TimeoutExpired:
        print("Re-encoding timed out.")
    except subprocess.CalledProcessError as e:
        print(f"Error during re-encoding: {e}\nOutput: {e.output}\nError: {e.stderr}")

    # Optional: Retry with different settings if needed



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

def download(url, folder, filepath=None, timeout=600):    
    """
    Robustly download a file from a given URL to the specified folder, automatically infering the file extension.
    
    Args:
        url (str):      The URL of the file to download.
        folder (str):   The folder where the downloaded file should be saved.
        filepath (str): (Optional) The path to the downloaded file. If None, the path will be inferred from the URL.
        
    Returns:
        filepath (Path): The path to the downloaded file.

    """
    try:
        if filepath is None:
            # Guess file extension from URL itself
            parsed_url_path = Path(url.split('/')[-1])
            ext = parsed_url_path.suffix
            
            # If extension is not in URL, then use Content-Type
            if not ext:
                response = requests.head(url, allow_redirects=True)
                content_type = response.headers.get('Content-Type')
                ext = mimetypes.guess_extension(content_type) or ''
            
            filename = parsed_url_path.stem + ext  # Append extension only if needed
            folder_path = Path(folder)
            filepath = folder_path / filename
        else:
            folder_path = Path(os.path.dirname(filepath))
        
        os.makedirs(folder_path, exist_ok=True)
        
        if os.path.exists(filepath):
            print(f"{filepath} already exists, skipping download..")
            return filepath
        
        print(f"Downloading {url} to {filepath}...")
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return filepath

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


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

                if 'images' in node_output:
                    outputs = []
                    for image in node_output['images']:
                        #image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        img_path = os.path.join(output_dir, image['subfolder'], image['filename'])
                        outputs.append(img_path)
                
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

    def get_workflow_output(self, args, verbose = False):
        # Dynamically choose the JSON file based on workflow type
        workflow_config_file = f"./custom_workflows/{args.mode}_api.json"
        print(f"Loading workflow config from {workflow_config_file}...")

        try:
            with open(workflow_config_file, 'r') as file:
                config = json.load(file)

            # Load the base workflow configuration
            input_config = f"./custom_workflows/{args.mode}_inputs.json"
            print(f"Loading input config from {input_config}...")
            with open(input_config, 'r') as file:
                input_config = json.load(file)

            # Populate the prompt dict using the config
            for key, value in input_config.items():
                arg_value = getattr(args, key, None)
                if arg_value is not None:
                    node_id = value["node_id"]
                    field = value["field"]
                    subfield = value["subfield"]
                    print(f"Overriding {node_id} {field} {subfield} -- with -- {arg_value}")
                    config[node_id][field][subfield] = arg_value

        except FileNotFoundError:
            print(f"{workflow_config_file} not found.")
            config = None
        except KeyError as e:
            print(f"Key error: {e}")
            config = None
        
        if verbose:
            # pretty print final config:
            print("------------------------------------------")
            print(json.dumps(config, indent=4, sort_keys=True))
            print("------------------------------------------")

        # start the process
        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(self.server_address, client_id))
        out_paths = self.get_output(ws, config, client_id)

        for node_id in out_paths:
            for out_path in out_paths[node_id]:
                return Path(out_path)

    def predict(
        self,
        render_mode: str = Input(
                    description="eden_vid2vid, eden_txt2vid or makeitrad", 
                    default = "eden_txt2vid",
                ),
        input_video_path: str = Input(
                    description="Load source video from file, url, or base64 string", 
                    default = None,
                ),
        prompt: str = Input(description="Prompt", default="the tree of life"),
        steps: int = Input(
            description="Steps",
            ge=10, le=40, default=20
        ),
        width: int = Input(
            description="Width", 
            ge=512, le=2048, default=768
        ),
        height: int = Input(
            description="Height", 
            ge=512, le=2048, default=768
        ),

        n_frames: int = Input(
            description="Total number of frames (mode==interpolate)",
            ge=16, le=264, default=32
        ),

        controlnet_type: str = Input(
            description="Controlnet type",
            default="qr_monster",
            choices=["canny-edge", "qr_monster"]
        ),
        controlnet_strength: float = Input(
            description="Strength of controlnet guidance", 
            ge=0.0, le=1.5, default=0.6
        ),

        guidance_scale: float = Input(
            description="Strength of text conditioning guidance", 
            ge=1, le=20, default=7.5
        ),
        negative_prompt: str = Input(description="Negative Prompt", default="nude, naked, text, watermark, low-quality, signature, padding, margins, white borders, padded border, moiré pattern, downsampling, aliasing, distorted, blurry, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, grainy, error, bad-contrast"),
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

        if input_video_path:
            # download video from url:
            video_path = download(input_video_path, "tmp_vids")
            # re-encode to .mp4 by default to avoid issues with the source video:
            input_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            reencode_video(video_path, input_video_path)
        elif render_mode == "vid2vid":
            raise ValueError("An input video is required for vid2vid mode!")

        # gather args from the input fields:
        args = {
            "input_video_path": input_video_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "n_frames": n_frames,
            "guidance_scale": guidance_scale,
            "mode": render_mode,
            "controlnet_type": controlnet_map[controlnet_type],
            "controlnet_strength": controlnet_strength,
            "steps": steps,
            "seed": seed,
        }
        args = AttrDict(args)

        # Run the ComfyUI job:
        try:
            output_path = self.get_workflow_output(args)
        except Exception as e:
            print(f"Error: {e}")
            output_path = None

        if output_path is None:
            if DEBUG_MODE:
                print(f'Returning {output_path} (DEBUG mode)')
                yield Path(output_path)
            else:
                output_path = Path(output_path)
                thumbnail_path = save_first_frame_to_tempfile(str(output_path))
                print(f'Returning {output_path} and {thumbnail_path}')
                yield CogOutput(files=[output_path], name=prompt, thumbnails=[Path(thumbnail_path)], attributes=None, progress=1.0, isFinal=True)
        else:
            print(f"output_path was None...")
            yield CogOutput(files=[], progress=1.0, isFinal=True)