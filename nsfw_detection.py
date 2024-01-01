import argparse
from typing import List

import tensorflow as tf
# Make sure TF doesnt allocate all the gpu memory:
gpus = tf.config.list_physical_devices('GPU')
tf_gpu_limit = 4096 # will force TF to only use 4GB of GPU memory
tf_gpu_limit = None # will trigger memory_growth (making TF only use what's needed)

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        if tf_gpu_limit is not None:
            tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=tf_gpu_limit)])
        else:
            tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# not sure if this helps/not:
#tf.keras.mixed_precision.set_global_policy('mixed_float16')

from absl import logging as absl_logging
tf.get_logger().setLevel('ERROR')
absl_logging.set_verbosity(absl_logging.ERROR)
import numpy as np
import sys, os

nsfw_repo_folder  = os.path.join("private-detector")
nsfw_model_folder = os.path.join("private-detector/models/bumble_nsfw_saved_model")

nsfw_model = tf.saved_model.load(nsfw_model_folder)
print("--- NSFW model loaded! ---")

sys.path.append(nsfw_repo_folder)
from private_detector.utils.preprocess import preprocess_for_evaluation

def read_image(filename: str) -> tf.Tensor:
    """
    Load and preprocess image for inference with the Private Detector

    Parameters
    ----------
    filename : str
        Filename of image

    Returns
    -------
    image : tf.Tensor
        Image ready for inference
    """
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)

    image = preprocess_for_evaluation(
        image,
        480,
        tf.float16
    )

    image = tf.reshape(image, -1)

    return image


def lewd_detection(image_paths: List[str], verbose = False) -> None:
    """
    Get predictions with a Private Detector model

    Parameters
    ----------
    image_paths : List[str]
        Path(s) to image to be predicted on
    """

    # if image_paths is a single string, convert to list
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # Make sure all image paths are strings:
    image_paths = [str(img_path) for img_path in image_paths]

    probs = []
    for image_path in image_paths:
        try:
            image = read_image(image_path)
            preds = nsfw_model([image])
            prob = tf.get_static_value(preds[0])[0]
            probs.append(np.round(prob, 3))
            if verbose:
                print(f'NSFW Prob: {100 * prob:.2f}% - {image_path}')
        except Exception as e:
            print(f'Error on nsfw-detection: {e} - {image_path}')
            probs.append(0.0)

    return probs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_paths',
        type=str,
        nargs='+',
        required=True,
        help='Paths to image paths to predict for'
    )

    args = parser.parse_args()
    lewd_detection(**vars(args), verbose = True)

