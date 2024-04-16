# don't push DEBUG_MODE = True to Replicate!
DEBUG_MODE = False
#DEBUG_MODE = True

import subprocess
import threading
import time
from cog import BasePredictor, BaseModel, Input, Path as cogPath
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
import io
import cv2
import tempfile
import subprocess
import os
import shlex

from nsfw_detection import lewd_detection

if DEBUG_MODE:
    debug_output_dir = "/src/tests/debug_output"
    if os.path.exists(debug_output_dir):
        shutil.rmtree(debug_output_dir)
    os.makedirs(debug_output_dir, exist_ok=True)

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

    compression_rate = 85  # Set compression quality (0-100)
    cv2.imwrite(temp_file.name, frame, [cv2.IMWRITE_JPEG_QUALITY, compression_rate])
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

    # Base FFmpeg command with scale filter to ensure even dimensions
    base_command = f"""
        ffmpeg -y -err_detect ignore_err -i "{input_file_path_str}" -vf "scale='2*trunc(iw/2)':'2*trunc(ih/2)':force_original_aspect_ratio=decrease"
        -c:v libx264 -preset fast
    """
    
    # Exclude audio codec for GIF files
    if not is_gif:
        base_command += ' -c:a aac'

    base_command += f' "{output_file_path}"'

    try:
        # Run the command with a timeout (e.g., 360 seconds)
        subprocess.run(shlex.split(base_command), check=True, timeout=360, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Re-encoding completed: {output_file_path}")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"Error or timeout during re-encoding. Copying original file to output. Error: {e}")
        shutil.copyfile(input_file_path_str, output_file_path)


def has_audio(file_path):
    """Check if a video file has an audio stream."""
    command = f"ffprobe -v error -select_streams a -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 {file_path}"
    result = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return len(result.stdout) > 0

def add_audio(generated_video, orig_video):
    """
    Check if the original video has audio. If so, add it to the generated video using ffmpeg.
    """
    # Check if the original video has an audio stream
    if not has_audio(orig_video):
        print("Original video has no audio stream.")
        return generated_video

    # Split the filename and extension
    file_root, file_ext = os.path.splitext(os.path.abspath(generated_video))
    output_file = f"{file_root}_with_audio{file_ext}"

    # Prepare the FFmpeg command for adding audio
    command = f"""
        ffmpeg -y -i "{generated_video}" -i "{orig_video}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest "{output_file}"
    """

    try:
        # Run the command with a timeout (e.g., 300 seconds)
        subprocess.run(shlex.split(command), check=True, timeout=300, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio added to video: {output_file}")
        return output_file
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"Error or timeout during adding audio. Error: {e}")
        return generated_video


class CogOutput(BaseModel):
    files: List[cogPath]
    name: Optional[str] = None
    thumbnails: Optional[List[cogPath]] = None
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

def download(url, folder, filepath=None, timeout=600, force_redownload = True):    
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
            parsed_url_path = cogPath(url.split('/')[-1])
            ext = parsed_url_path.suffix
            
            # If extension is not in URL, then use Content-Type
            if not ext:
                response = requests.head(url, allow_redirects=True)
                content_type = response.headers.get('Content-Type')
                ext = mimetypes.guess_extension(content_type) or ''
            
            filename = parsed_url_path.stem + ext  # Append extension only if needed
            folder_path = cogPath(folder)
            filepath = folder_path / filename
            filepath = str(filepath.absolute())
        else:
            folder_path = os.path.dirname(filepath)
        
        os.makedirs(folder_path, exist_ok=True)
        
        if os.path.exists(filepath) and not force_redownload:
            print(f"{filepath} already exists, skipping download..")
            return str(filepath)
        
        print(f"Downloading {url} to {filepath}...")
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return str(filepath)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def format_prompt(prompt, n_frames):
    prompt_list = prompt.split('|')
    n_frames_per_prompt = n_frames // len(prompt_list)

    formatted_prompt = ""
    for i, p in enumerate(prompt_list):
        frame = str(i * n_frames_per_prompt)
        formatted_prompt += f"\"{frame}\" : \"{p}\",\n"

    # Removing the last comma and newline
    formatted_prompt = formatted_prompt.rstrip(',\n')
    print("Final prompt string:")
    print(formatted_prompt)

    return prompt_list, formatted_prompt
    

class Predictor(BasePredictor):

    GENERATOR_OUTPUT_TYPE = cogPath if DEBUG_MODE else CogOutput

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

    def get_output(self, ws, config, client_id):
        print("Sending job to ComfyUI server...")
        prompt_id = self.queue_prompt(config, client_id)['prompt_id']
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
                    # Make sure to escape all tabs and newlines for the JSON:
                    arg_value = arg_value.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
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
        print("Running pipeline...")
        out_paths = self.get_output(ws, config, client_id)

        # Grab the pipeline output:
        for node_id in out_paths:
            for out_path in out_paths[node_id]:
                
                if args.input_video_path and args.mode == "vid2vid": # the input was a video and it had audio, add it back to the output video:
                    out_path = add_audio(out_path, args.input_video_path)

                if out_path.endswith('.png'): # convert .png to .jpg with 95% quality:
                    img = Image.open(out_path).convert("RGB")
                    os.remove(out_path)
                    out_path = out_path.replace('.png', '.jpg')
                    img.save(out_path, quality=95)

                return cogPath(out_path)

    def predict(
        self,
        mode: str = Input(
                    description="txt2vid, img2vid, vid2vid, upscale, txt2img, inpaint, makeitrad", 
                    default = "txt2vid",
                ),
        text_input: str = Input(description="prompt", default=None),
        interpolation_texts: str = Input(description="| separated list of prompts for txt2vid)", default=None),
        input_images: str = Input(
                    description="Input image(s) for various endpoints. Load-able from file, url, or base64 string, (urls separated by pipe symbol)", 
                    default = None,
                ),
        style_images: str = Input(
                    description="Input style image(s) (for IP_adapter) for various endpoints. Load-able from file, url, or base64 string, (urls separated by pipe symbol)", 
                    default = None,
                ), 
        mask_images: str = Input(
                    description="Input mask image(s) for various endpoints. Load-able from file, url, or base64 string, (urls separated by pipe symbol)", 
                    default = None,
                ),
        input_video_path: str = Input(
                    description="For vid2vid. Load source video from file, url, or base64 string", 
                    default = None,
                ),
        steps: int = Input(
            description="Steps",
            ge=10, le=40, default=25
        ),
        width: int = Input(
            description="Width", 
            ge=512, le=3840, default=1280
        ),
        height: int = Input(
            description="Height", 
            ge=512, le=3840, default=1280
        ),
        n_frames: int = Input(
            description="Total number of frames (txt2vid, vid2vid, img2vid)",
            ge=16, le=264, default=40
        ),
        ip_adapter_weight: float = Input(
            description="Strenght of the IP_adapter style",
            ge=0.0, le=2.0, default=0.65
        ),
        motion_scale: float = Input(
            description="Motion scale (AnimateDiff)",
            ge=0.0, le=2.0, default=1.1
        ),
        n_samples: int = Input(
            description="batch size",
            ge=1, le=4, default=1
        ),
        control_method: str = Input(
            description="Shape Control method (coarse usually gives nicer results, fine is more precise to the input video)",
            default="coarse",
            choices=["coarse", "fine"]
        ),
        controlnet_strength: float = Input(
            description="Strength of controlnet guidance", 
            ge=0.0, le=1.5, default=0.85
        ),
        denoise_strength: float = Input(
            description="How much denoising to apply (1.0 = start from full noise, 0.0 = return input image)", 
            ge=0.0, le=1.0, default=1.0
        ),
        blend_value: float = Input(
            description="Blend factor (weight of the first image vs the second)", 
            ge=0.0, le=1.0, default=0.5
        ),
        loop: bool = Input(
            description="Try to make a loopable video",
            default=False
        ),
        guidance_scale: float = Input(
            description="Strength of text conditioning guidance", 
            ge=1, le=20, default=7.5
        ),
        negative_prompt: str = Input(
            description="Negative Prompt", 
            default="nude, naked, text, watermark, low-quality, signature, padding, margins, white borders, padded border, moiré pattern, downsampling, aliasing, distorted, blurry, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, grainy, error, bad-contrast"
        ),
        seed: int = Input(description="Sampling seed, leave Empty for Random", default=None),

    ) -> Iterator[GENERATOR_OUTPUT_TYPE]:
        t_start = time.time()
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(3), "big")

        controlnet_map = {
            "canny-edge": "control_sd15_canny.safetensors",
            "qr_monster": "control_v1p_sd15_qrcode_monster.safetensors",
        }
        if mode == "txt2vid":
            if not interpolation_texts:
                raise ValueError("You forgot to enter the interpolation texts!")

        if mode == "makeitrad":
            if ("embedding:makeitrad_embeddings" not in text_input) and ("embedding:indoor-outdoor_embeddings" not in text_input):
                raise ValueError("You forgot to trigger the LoRa concept, add 'embedding:makeitrad_embeddings' or 'embedding:indoor-outdoor_embeddings' somewhere in the prompt!")

        if mode in ["vid2vid", "blend", "upscale"]: # these modes use a 2x_upscaler at the end:
            width = width // 2
            height = height // 2
        if mode in ["img2vid", "txt2vid"]:
            width = width // (2 * 1.5)
            height = height // (2 * 1.5)

        if interpolation_texts:  # For now, just equally space the prompts!
            text_input = interpolation_texts
            prompt_list, interpolation_texts = format_prompt(interpolation_texts, n_frames)
        
        if input_video_path:
            downloaded_video_path = download(input_video_path, "tmp_vids")
            # re-encode to .mp4 by default to avoid issues with the source video:
            input_video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            reencode_video(downloaded_video_path, input_video_path)
        elif mode == "vid2vid":
            raise ValueError("An input video/gif is required for vid2vid mode!")


        def prep_images(image_url_str):
            image_paths = []
            if image_url_str:
                for image_url in image_url_str.split("|"):
                    print(f"Downloading {image_url}...")
                    image_url = download(image_url, "tmp_imgs", force_redownload = True)
                    image_paths.append(image_url)

            return image_paths

        # download the images from the given urls:
        input_image_paths = prep_images(input_images)
        style_image_paths = prep_images(style_images)
        mask_image_paths  = prep_images(mask_images)

        # Handle some custom cases:
        if mode in ["vid2vid"]:
            print(f"in vid2vid mode!")
            print("len(style_image_paths):", len(style_image_paths))
            if len(style_image_paths) == 0:
                raise ValueError(f"At least one style image is required for mode {mode}!")

            if len(style_image_paths) == 1: # if there's only one style img, just copy that one to the second!
                print("Setting the second style image to the first one...")
                style_image_paths.append(style_image_paths[0])

        if mode in ["upscale", "img2vid", "blend", "inpaint"] and len(input_image_paths) == 0:
            raise ValueError(f"An input image is required for mode {mode}!")

        if len(mask_image_paths) == 0: # if no mask was provided, just use a default all white mask:
            mask_image_paths.append("/src/white_mask.png")

        if mode == "upscale":
            # the UI form only exposes 'width' as 'Resolution' so just copy it over to height in this mode
            height = width

        if text_input is None and mode in ["img2vid", "vid2vid", "inpaint"]:
            # in ComfyUI an empty string should be skipped by the pipeline flow and trigger CLIP-interrogator instead
            text_input = "" 

        # Default settings
        input_video_frame_rate = 8 # input fps
        RIFE_multiplier        = 3 # output RIFE framerate multiplier

        if mode == "vid2vid":
            # check if there's at least 2 seconds of video:
            cap = cv2.VideoCapture(input_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            n_video_seconds = total_n_frames / fps
            cap.release()

            if total_n_frames < 1.5 * fps:
                raise ValueError(f"The input video/GIF must be at least 1.5 seconds long for vid2vid mode to work (The given input is only {n_video_seconds:.2f} seconds long)!")

            # If there's only a small amount of total seconds, we increase the diffusion framerate a bit:
            if n_video_seconds < 4:
                input_video_frame_rate = 12 # input fps
                RIFE_multiplier        = 2 # output RIFE framerate multiplier

        if len(style_image_paths) == 0: # If no input imgs are provided, set the ip_adapter weight to 0:
            print("No input images provided, setting ip_adapter_weight to 0.0..")
            ip_adapter_weight = 0.0
            if mode == "txt2vid": # the pipe breaks if there's no style img, so we set a default one:
                style_image_paths = ["/src/white_mask.png"]

        print("---------------")
        print("input_image_paths:", input_image_paths)
        print("style_image_paths:", style_image_paths)
        print("mask_image_paths:", mask_image_paths)

        # gather args from the input fields:
        args = {
            "input_video_path": input_video_path,
            "input_image_path1": input_image_paths[0] if len(input_image_paths) > 0 else None,
            "input_image_path2": input_image_paths[1] if len(input_image_paths) > 1 else None,
            "mask_image_path": mask_image_paths[0],
            "style_image_path1": style_image_paths[0] if len(style_image_paths) > 0 else None,
            "style_image_path2": style_image_paths[1] if len(style_image_paths) > 1 else None,
            "text_input": text_input,
            "interpolation_texts": interpolation_texts,
            "negative_prompt": negative_prompt,
            "force_rate": input_video_frame_rate,
            "RIFE_multiplier": RIFE_multiplier,
            "ip_adapter_weight": ip_adapter_weight,
            "width": width,
            "height": height,
            "n_frames": n_frames,
            "n_frames2": n_frames, # temporary hack for comfy_txt2vid where this field is needed twice
            "motion_scale": motion_scale,
            "guidance_scale": guidance_scale,
            "mode": mode,
            "denoise_strength": denoise_strength,
            #"controlnet_type": controlnet_map[controlnet_type],
            "control_method": control_method,
            "controlnet_strength": controlnet_strength,
            "steps": steps,
            "seed": seed,
            "loop": loop,
        }
        args = AttrDict(args)

        print('------------------------------------------')
        print(f"Running mode {mode} with the following args:")
        print(args)
        print('------------------------------------------')

        if not text_input:
            text_input = mode
        
        output_paths = []
        try: # Run the ComfyUI job:
            print(f"Running {mode} comfy pipeline {n_samples} times!")
            for i in range(n_samples):
                output_path = self.get_workflow_output(args)
                output_paths.append(str(output_path))
                args.seed += 1

        except Exception as e:
            print(f"Error in self.get_workflow_output(): {e}")
            output_paths = [""] * n_samples

        print("------------------------------------------")
        print("Pipeline finished!")
        print(f"Output paths: {output_paths}")
        print("------------------------------------------")

        if DEBUG_MODE:
            prediction_name = f"{mode}_{seed}_{n_samples}"
            os.makedirs(debug_output_dir, exist_ok=True)

            for index in range(n_samples):
                    save_path = os.path.join(debug_output_dir, prediction_name + f"_{index}.jpg")
                    Image.new("RGB", (512, 512), "black").save(save_path)

            print(f'Returning {output_paths} (DEBUG mode)')
            # Save the outputs to disk:
            for index, output_path in enumerate(output_paths):
                if output_path != "":
                    shutil.copy(output_path, os.path.join(debug_output_dir, prediction_name + f"_{index}.jpg"))

            yield [cogPath(output_path) for output_path in output_paths]
        else:
            thumbnail_paths = [save_first_frame_to_tempfile(output_path) if output_path != "" else "" for output_path in output_paths]

            attributes = {}
            attributes['n_samples']   = n_samples
            attributes['nsfw_scores'] = lewd_detection(thumbnail_paths)
            attributes['job_time_seconds'] = time.time() - t_start
            attributes['seeds']       = [args.seed - n_samples + i for i in range(n_samples)]

            print(f'Returning {output_paths} and {thumbnail_paths}')
            print(f"text_input: {text_input}")
            print(f"attributes: {attributes}")

            output_paths    = [cogPath(output_path) for output_path in output_paths]
            thumbnail_paths = [cogPath(thumbnail_path) for thumbnail_path in thumbnail_paths]

            yield CogOutput(files=output_paths, name=text_input, thumbnails=thumbnail_paths, attributes=attributes, progress=1.0, isFinal=True)
