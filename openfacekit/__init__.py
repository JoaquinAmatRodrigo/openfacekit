__version__ = "0.2.0"

from .face_recognition import (
    FaceRecognizer,
    ReferenceEmbeddings
)

from .utils import (
    convert_to_matplotlib_rgb,
    convert_to_opencv_bgr,
    download_opencv_yunet_model,
)

__all__ = [
    "FaceRecognizer",
    "ReferenceEmbeddings", 
    "convert_to_matplotlib_rgb",
    "convert_to_opencv_bgr",
    "download_opencv_yunet_model",
]