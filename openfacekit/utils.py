################################################################################
#                                  utils                                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import os
from urllib.request import urlretrieve
import numpy as np
import cv2
import torch

def convert_to_matplotlib_rgb(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert a face tensor to RGB format for matplotlib display.
    The input tensor is expected to be in the range [-1, 1] and have shape
    [3, height, width]. The output is a numpy array in RGB format with values
    in the range [0, 1]. The output can be directly displayed with matplotlib.

    Parameters
    ----------
    image: torch.Tensor, np.ndarray
        Tensor or numpy array representing the face image. It should have shape
        [3, height, width] and values in the range [-1, 1].

    Returns
    -------
    np.ndarray
        Numpy array in RGB format with values in the range [0, 1]. The shape is
        [height, width, 3], suitable for display with matplotlib.

    """
    if not isinstance(image, (torch.Tensor, np.ndarray)):
        raise ValueError(
            f"`image` must be a `torch.Tensor` or `np.ndarray`. Got {type(image)}."
        )

    if image.shape[0] != 3:
        raise ValueError(f"Expected image shape [3, H, W], but got {image.shape}")

    if isinstance(image, np.ndarray):
        image = np.clip((image + 1) / 2, 0, 1).transpose(1, 2, 0)
    else:
        image = image.add(1).div(2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    return image


def convert_to_opencv_bgr(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert a face tensor to BGR format for OpenCV display.

    The input tensor is expected to be in the range [-1, 1] and have shape
    [3, height, width]. The output is a numpy array in BGR format with values
    in the range [0, 255]. The output can be directly used with OpenCV.

    Parameters
    ----------
    image: torch.Tensor, np.ndarray
        Tensor or numpy array representing the face image. It should have shape
        [3, height, width] and values in the range [-1, 1].

    Returns
    -------
    np.ndarray
        Numpy array in BGR format with values in the range [0, 255]. The shape is
        [height, width, 3], suitable for display with OpenCV.

    """
    if not isinstance(image, (torch.Tensor, np.ndarray)):
        raise ValueError(
            f"`image` must be a `torch.Tensor` or `np.ndarray`. Got {type(image)}."
        )

    if image.shape[0] != 3:
        raise ValueError(f"Expected image shape [3, H, W], but got {image.shape}")

    if isinstance(image, np.ndarray):
        image = np.clip((image + 1) * 127.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = (
            image.add(1).mul(127.5).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
        )
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def frame_generator(cap):
    """
    Generator function to yield frames from a video capture object.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

def download_opencv_yunet_model(
        save_path: None | str = None,
        model_url: None | str = None,
        ) -> str:
    """
    Downloads the OpenCV Yunet model weights file. If no `save_path` is provided,
    the model is saved in the current working directory with the default name.
    If no `model_url` is provided, the default OpenCV Yunet model URL is used: 
    https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx?download=true

    Parameters
    ----------
    save_path: str
        Path to save the downloaded model weights file.
    model_url: str
        URL to download the model weights file.

    Returns
    -------
    None
    """

    if model_url is None:
        model_url = (
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
            "face_detection_yunet_2023mar.onnx?download=true"
        )
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "face_detection_yunet.onnx")
    if not os.path.exists(save_path):
        print("Downloading OpenCV Yunet model...")
        urlretrieve(model_url, save_path)
        print(f"Model downloaded and saved to {save_path}")
    else:
        print(f"Model already exists at {save_path}")
    return save_path