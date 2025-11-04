################################################################################
#                              face_recognition                                #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import logging
import os
import io
import glob
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2 as cv
import facenet_pytorch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from joblib import load, dump
from tqdm import tqdm

from .utils import (
    convert_to_matplotlib_rgb,
    convert_to_opencv_bgr,
    frame_generator
)


logging.basicConfig(
    format="%(asctime)-5s %(name)-10s %(levelname)-5s %(message)s",
    level=logging.WARNING,
)

class FaceRecognizer:
    """
    Class for face detection and identification. This class wraps the MTCNN or OpenCV Yunet and
    InceptionResnetV1 models for easy face detection and recognition in images and videos.

    Parameters
    ----------
    detector: facenet_pytorch.models.mtcnn.MTCNN, cv.FaceDetectorYN, str, default "MTCNN"
        Model used to detect faces in images:
        - If "MTCNN": initializes a MTCNN model from the `facenet_pytorch` library.
        - If "OpenCV_Yunet": initializes a Yunet model from OpenCV. The path to the model
          weights must be indicated using the argument `model_path`.
        - If a `MTCNN` or `cv.FaceDetectorYN` instance is provided, it uses that model.
    opencv_yunet_model_path: str, default `None`
        Path to the OpenCV Yunet model weights file. This parameter is only used if
        `detector` is set to "OpenCV_Yunet". User can use function `download_opencv_yunet_model`
        to download the model weights.
    encoder: facenet_pytorch.models.inception_resnet_v1.InceptionResnetV1, default `None`
        InceptionResnetV1 model used to obtain numerical embeddings of faces. If `None`, a new one
        is initialized.
    min_face_size: int, default `20`
        Minimum size that faces must have to be detected by the MTCNN network.
    thresholds_mtcnn: list, default `[0.6, 0.7, 0.7]`
        Detection thresholds for each of the three networks forming the MTCNN detector.
    min_confidence_detector: float, default `0.5`
        Minimum confidence (probability) required for a detected face to be included
        in the results. This only applies to the MTCNN detector since the OpenCV Yunet
        does not provide confidence scores. If the Yunet detector is used, all detected faces
        are included.
    similarity_threshold: float, default 0.5
        Minimum similarity required between embeddings to assign an identity.  Otherwise, the label
        `None` is assigned.
    similarity_metric: str, default 'euclidean'
        Metric used to calculate the similarity between embeddings. Options are:
        - 'euclidean': similarity is defined as 1 - (euclidean_distance / 2),
        normalized to [0, 1], where 1 indicates maximum similarity.
        - 'cosine': cosine similarity between embeddings. Ranges from -1 to 1,
        where 1 indicates maximum similarity.
    device: str, default `None`
        Device where the models are executed. If `None`, it uses the first available GPU
        or CPU if no GPU is available.
    verbose: bool, default `False`
        Display process information on the screen.

    Attributes
    ----------
    detector: facenet_pytorch.models.mtcnn.MTCNN | cv.FaceDetectorYN
        Model used to detect faces in images.
    encoder: facenet_pytorch.models.inception_resnet_v1.InceptionResnetV1
        InceptionResnetV1 model used to obtain numerical embeddings of faces.
    min_face_size: int
        Minimum size that faces must have to be detected by the MTCNN network.
    thresholds: list
        Detection thresholds for each of the three networks forming the MTCNN detector.
    min_confidence_detector: float
        Minimum confidence (probability) required for a detected face to be included
        in the results. This only applies to the MTCNN detector since the OpenCV Yunet
        does not provide confidence scores. If the Yunet detector is used, all detected faces
        are included.
    similarity_threshold: float, default 0.5
        Minimum similarity required between embeddings to assign an identity. Otherwise, the label
        `None` is assigned.
    similarity_metric: str, default 'euclidean'
        Metric used to calculate the similarity between embeddings. Options are:
        - 'euclidean': similarity is defined as 1 - (euclidean_distance / 2),
        normalized to [0, 1], where 1 indicates maximum similarity.
        - 'cosine': cosine similarity between embeddings. Ranges from -1 to 1,
        where 1 indicates maximum similarity.
    reference_embeddings: dict, default None
        Dictionary containing reference embeddings. The key represents the identity
        of the person, and the value is the embedding of their face.
    verbose: bool
        Display process information on the screen.
    device: str
        Device where the models are executed.

    """

    def __init__(
        self,
        detector: facenet_pytorch.models.mtcnn.MTCNN | cv.FaceDetectorYN | str = "MTCNN",
        opencv_yunet_model_path: str | None = None,
        encoder: (
            facenet_pytorch.models.inception_resnet_v1.InceptionResnetV1 | None
        ) = None,
        min_face_size: int = 20,
        thresholds: list[float] = [0.6, 0.7, 0.7],
        min_confidence_detector: float = 0.5,
        similarity_threshold: float = 0.5,
        similarity_metric: str = "cosine",
        keep_all: bool = True,
        device: str | None = None,
        verbose: bool = False,
    ):

        if not isinstance(thresholds, list) or len(thresholds) != 3:
            raise ValueError(
                f"`thresholds` must be a list of three float values. Got {thresholds}."
            )

        if not (0 <= min_confidence_detector <= 1):
            raise ValueError(
                f"`min_confidence_detector` must be between 0 and 1. Got {min_confidence_detector}."
            )

        if not isinstance(min_face_size, int) or min_face_size <= 0:
            raise ValueError(
                f"`min_face_size` must be a positive integer. Got {min_face_size}."
            )

        if not (0 <= similarity_threshold <= 1):
            raise ValueError(
                f"`similarity_threshold` must be between 0 and 1. Got {similarity_threshold}."
            )

        if similarity_metric not in ["cosine", "euclidean"]:
            raise ValueError(
                f"`similarity_metric` must be 'cosine' or 'euclidean'. Got {similarity_metric}."
            )

        if not isinstance(keep_all, bool):
            raise ValueError(f"`keep_all` must be a boolean. Got {type(keep_all)}.")

        if device and not isinstance(device, str):
            raise ValueError(f"`device` must be a string. Got {type(device)}.")

        if not device:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")

        if not isinstance(detector, (str, MTCNN, cv.FaceDetectorYN)):
            raise ValueError(
                f"`detector` must be 'MTCNN', 'OpenCV', or an instance of MTCNN or cv.FaceDetectorYN. "
                f"Got {type(detector)}."
            )
        if detector == "MTCNN":
            logging.info("Initializing MTCNN detector")
            detector = MTCNN(
                keep_all=keep_all,
                min_face_size=min_face_size,
                thresholds=thresholds,
                post_process=True,
                device=device,
            )
        if detector == "OpenCV_Yunet":
            if opencv_yunet_model_path is None:
                raise ValueError(
                    "`opencv_yunet_model_path` must be provided when `detector` is 'OpenCV_Yunet'."
                    "User can use function `download_opencv_yunet_model` to download the model weights."
                )
            logging.info("Initializing OpenCV Yunet detector")
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                backend_id = cv.dnn.DNN_BACKEND_CUDA
                target_id = cv.dnn.DNN_TARGET_CUDA
            else:
                backend_id = cv.dnn.DNN_BACKEND_OPENCV
                target_id = cv.dnn.DNN_TARGET_CPU
            detector = cv.FaceDetectorYN.create(
                model = opencv_yunet_model_path,
                config = "",
                input_size = (320, 320),
                score_threshold = 0.5,
                backend_id=backend_id,
                target_id=target_id
            )

        if encoder is None:
            logging.info("Initializing InceptionResnetV1 encoder")
            encoder = InceptionResnetV1(pretrained="vggface2", classify=False).eval()
            encoder = encoder.to(device)

        self.detector = detector
        self.encoder = encoder
        self.device = device
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.min_confidence_detector = min_confidence_detector
        self.similarity_threshold = similarity_threshold
        self.similarity_metric = similarity_metric
        self.keep_all = keep_all
        self.verbose = verbose
        self.reference_embeddings_values = None
        self.reference_embeddings = None

    def __repr__(self):
        text = (
            f"--------------\n",
            f"FaceRecognizer\n",
            f"--------------\n",
            f"Detector type: {type(self.detector).__name__}\n",
            f"Encoder type: {type(self.encoder).__name__}\n",
            f"Device: {self.device}\n",
            f"Number of reference identities: {len(self.reference_embeddings) if self.reference_embeddings is not None else 0}\n",
            f"Similarity metric: {self.similarity_metric}\n",
            f"Similarity threshold: {self.similarity_threshold}\n",
            f"Minimum confidence detector: {self.min_confidence_detector}\n",
        )
        return "".join(text)

    def detect_bboxes(
        self,
        image: PIL.Image.Image | np.ndarray,
        fix_bbox: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect the position of faces in an image using an MTCNN detector or OpenCV Yunet.

        Parameters
        ----------
        image: PIL.Image, np.ndarray
            PIL Image or numpy array representing the image.
        fix_bbox : bool, default: True
            Adjusts the dimensions of the bounding boxes so they do not exceed the
            dimensions of the image. This avoids issues when trying to represent
            bounding boxes of faces that are at the edge of the image.

        Returns
        ----------
        boxes: numpy.ndarray
            Numpy array with the bounding boxes of each detected face. Each bounding
            box is an array formed by 4 values that define the coordinates of the
            top-left corner and the bottom-right corner.

                 (box[0],box[1])------------
                        |                  |
                        |                  |
                        |                  |
                        ------------(box[0],box[1])

            The bounding boxes returned by the ``MTCNN`` detector are defined by
            `float` values. This poses a problem for subsequent representation with
            matplotlib, so they are converted to `int` type.
        probs: numpy.ndarray
            Numpy array with the confidence (probability) of each detected face.
            Each value is a float between 0 and 1, where 1 indicates maximum confidence
            that the detected face is indeed a face. A value of 0 indicates no confidence
            that the detected face is a face. The values are used to filter out faces
            that do not meet the minimum confidence threshold specified by the
            `min_confidence_detector` parameter.

        """

        if not isinstance(image, (np.ndarray, PIL.Image.Image)):
            raise Exception(
                f"`image` must be `np.ndarray, PIL.Image`. Got {type(image)}."
            )

        if isinstance(self.detector, MTCNN):
            bboxes, probs = self._detect_bboxes_mtcnn(image, fix_bbox=fix_bbox)
        else:
            bboxes, probs = self._detect_bboxes_cv_younet(image, fix_bbox=fix_bbox)

        return bboxes, probs

    def _detect_bboxes_mtcnn(
        self,
        image: PIL.Image.Image | np.ndarray,
        fix_bbox: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect the position of faces in an image using an MTCNN detector.

        Parameters
        ----------
        image: PIL.Image, np.ndarray
            PIL Image or numpy array representing the image. Color channels must be in RGB format.
        fix_bbox : bool, default: True
            Adjusts the dimensions of the bounding boxes so they do not exceed the
            dimensions of the image. This avoids issues when trying to represent
            bounding boxes of faces that are at the edge of the image.

        Returns
        ----------
        boxes: numpy.ndarray
            Numpy array with the bounding boxes of each detected face. Each bounding
            box is an array formed by 4 values that define the coordinates of the
            top-left corner and the bottom-right corner.

                 (box[0],box[1])------------
                        |                  |
                        |                  |
                        |                  |
                        ------------(box[0],box[1])

            The bounding boxes returned by the ``MTCNN`` detector are defined by
            `float` values. This poses a problem for subsequent representation with
            matplotlib, so they are converted to `int` type.
        probs: numpy.ndarray
            Numpy array with the confidence (probability) of each detected face.
            Each value is a float between 0 and 1, where 1 indicates maximum confidence
            that the detected face is indeed a face. A value of 0 indicates no confidence
            that the detected face is a face. The values are used to filter out faces
            that do not meet the minimum confidence threshold specified by the
            `min_confidence_detector` parameter.

        """

        if isinstance(image, PIL.Image.Image):
            image = np.array(image).astype(np.uint8)

        bboxes, probs = self.detector.detect(image, landmarks=False)

        if bboxes is not None and len(bboxes) > 0:

            valid_indices = probs > self.min_confidence_detector
            bboxes_valid = bboxes[valid_indices]
            probs_valid = probs[valid_indices]

            if fix_bbox:
                bboxes_valid = np.clip(
                    bboxes_valid,
                    [0, 0, 0, 0],
                    [image.shape[1], image.shape[0], image.shape[1], image.shape[0]],
                )
                bboxes_valid = bboxes_valid.astype(int)

        else:
            bboxes_valid = np.array([])
            probs_valid = np.array([])

        if self.verbose:
            print("----------------")
            print("Scanned image")
            print("----------------")
            print(f"Detected faces: {len(bboxes)}")
            print(f"Detected faces with minimum confidence: {len(bboxes_valid)}")
            print(f"Bounding box correction applied: {fix_bbox}")
            print(f"Bounding box coordinates: {bboxes_valid.tolist()}")
            print(f"Bounding box confidence: {probs_valid.tolist()}")
            print("")

        return bboxes_valid, probs_valid

    def _detect_bboxes_cv_younet(
        self,
        image: PIL.Image.Image | np.ndarray,
        fix_bbox: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect the position of faces in an image using an MTCNN detector.

        Parameters
        ----------
        image: PIL.Image, np.ndarray
            PIL Image or numpy array representing the image. Color channels must be in BGR format.
        fix_bbox : bool, default: True
            Adjusts the dimensions of the bounding boxes so they do not exceed the
            dimensions of the image. This avoids issues when trying to represent
            bounding boxes of faces that are at the edge of the image.

        Returns
        ----------
        boxes: numpy.ndarray
            Numpy array with the bounding boxes of each detected face. Each bounding
            box is an array formed by 4 values that define the coordinates of the
            top-left corner and the bottom-right corner.

                 (box[0],box[1])------------
                        |                  |
                        |                  |
                        |                  |
                        ------------(box[0],box[1])
        probs: numpy.ndarray
            Confidence (probability) of each detected face. Each value is a float between 0 and 1,
            where 1 indicates maximum confidence that the detected face is indeed a face. A value of 0
            indicates no confidence that the detected face is a face. The values are used to filter out
            faces that do not meet the minimum confidence threshold specified by the
            `min_confidence_detector` parameter.

        """

        if isinstance(image, PIL.Image.Image):
            image = np.array(image).astype(np.uint8)
            image = image[:, :, ::-1]  # RGB → BGR

        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, bboxes_scores = self.detector.detect(image)

        if bboxes_scores is not None and len(bboxes_scores) > 0:
            bboxes = np.array([
                np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]]).astype(int)
                for box in bboxes_scores
            ])
            probs = np.array([box[4] for box in bboxes_scores]).astype(float)
            valid_indices = probs > self.min_confidence_detector
            bboxes_valid = bboxes[valid_indices]
            probs_valid = probs[valid_indices]

            if fix_bbox:
                bboxes_valid = np.clip(
                    bboxes_valid,
                    [0, 0, 0, 0],
                    [image.shape[1], image.shape[0], image.shape[1], image.shape[0]],
                )
                bboxes_valid = bboxes_valid.astype(int)

        else:
            bboxes_valid = np.array([])
            probs_valid = np.array([])

        if self.verbose:
            print("----------------")
            print("Scanned image")
            print("----------------")
            print(f"Detected faces: {len(bboxes)}")
            print(f"Detected faces with minimum confidence: {len(bboxes_valid)}")
            print(f"Bounding box correction applied: {fix_bbox}")
            print(f"Bounding box coordinates: {bboxes_valid.tolist()}")
            print(f"Bounding box confidence: {probs_valid.tolist()}")
            print("")

        return bboxes_valid, probs_valid
        

    def display_bounding_boxes(
        self,
        image: PIL.Image.Image | np.ndarray,
        bboxes: np.ndarray,
        identities: list | None = None,
        save_path: str | None = None,
        return_image: bool = False,
    ) -> None:
        """
        Display the original image with bounding boxes of detected faces using matplotlib.
        If identities are provided, they are displayed above each bounding box.

        Parameters
        ----------
        image: PIL.Image, np.ndarray
            `PIL Image` or `numpy array` representing the image.
        bboxes: np.ndarray
            Numpy array with bounding boxes of faces present in the image.
            Each bounding box is an array of 4 values defining the coordinates of
            the top-left corner and the bottom-right corner.

                 (box[0],box[1])------------
                        |                  |
                        |                  |
                        |                  |
                        ------------(box[2],box[3])
        identities: list, default None
            Identity associated with each bounding box. Must have the same number
            of elements as `bboxes` and be aligned such that `identities[i]`
            corresponds to `bboxes[i]`.
        save_path: str, default None
            Path to save the output image with bounding boxes.
        return_image: bool, default False
            If `True`, the function returns the modified image with bounding boxes.

        Returns
        -------
        None

        """
        if not isinstance(image, (np.ndarray, PIL.Image.Image)):
            raise Exception(
                f"`image` must be `np.ndarray, PIL.Image`. Got {type(image)}."
            )

        if identities is not None and len(bboxes) != len(identities):
            raise Exception(
                "`identities` must have the same number of elements as `bboxes`."
            )

        if not identities:
            identities = [None] * len(bboxes)

        if isinstance(image, PIL.Image.Image):
            image = np.array(image).astype(np.float32) / 255

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")

        def create_rectangle(bbox, color):
            return plt.Rectangle(
                xy=(bbox[0], bbox[1]),
                width=bbox[2] - bbox[0],
                height=bbox[3] - bbox[1],
                linewidth=1,
                edgecolor=color,
                facecolor="none",
            )

        for bbox, identity in zip(bboxes, identities):
            color = "lime" if identity else "red"
            rect = create_rectangle(bbox, color)
            ax.add_patch(rect)

            if identity:
                ax.text(
                    x=bbox[0], y=bbox[1] - 10, s=identity, fontsize=10, color="lime"
                )

        if save_path:
            fig.savefig(save_path)
            logging.info(f"Bounding boxes saved to {save_path}")

        if return_image:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)  # Close the figure to avoid memory leaks
            buf.seek(0)
            return Image.open(buf)
        else:
            plt.show()

    def display_bounding_boxes_cv(
        self,
        image: PIL.Image.Image | np.ndarray,
        bboxes: np.ndarray,
        identities: list | None = None,
        window_name: str = "Frame",
        save_path: str | None = None,
        return_image: bool = False,
    ) -> None:
        """
        Display the original image with bounding boxes of detected faces using OpenCV.
        If identities are provided, they are displayed above each bounding box.
        This function cannot be used inside a Jupyter notebook.

        Parameters
        ----------
        image: PIL.Image | np.ndarray
            `PIL Image` or `numpy array` representing the image.
        bboxes: np.ndarray
            Numpy array with bounding boxes of faces present in the image.
            Each bounding box is an array of 4 values defining the coordinates of
            the top-left corner and the bottom-right corner.

                 (box[0],box[1])------------
                        |                  |
                        |                  |
                        |                  |
                        ------------(box[2],box[3])
        identities: list | None, default None
            Identity associated with each bounding box. Must have the same number
            of elements as `bboxes` and be aligned such that `identities[i]`
            corresponds to `bboxes[i]`.
        window_name: str, default "Frame"
            Name of the pop-up window opened by cv.imshow(). If `None`, the image
            is returned but not displayed in a window.
        save_path: str | None, default None
            Path to save the output image with bounding boxes.
        return_image: bool, default False
            If True, the processed image is returned instead of being displayed.

        Returns
        -------
        None

        """
        if not isinstance(image, (np.ndarray, PIL.Image.Image)):
            raise Exception(
                f"`image` must be `np.ndarray`, `PIL.Image`. Got {type(image)}."
            )

        if identities is not None and len(bboxes) != len(identities):
            raise Exception(
                "`identities` must have the same number of elements as `bboxes`."
            )

        if identities is None:
            identities = [None] * len(bboxes)

        if isinstance(image, PIL.Image.Image):
            image = np.array(image).astype(np.float32) / 255

        if image.shape[-1] == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        def add_rectangle_and_text(img, bbox, identity):
            color = (0, 255, 0) if identity else (255, 0, 0)
            cv.rectangle(
                img=img,
                pt1=(bbox[0], bbox[1]),
                pt2=(bbox[2], bbox[3]),
                color=color,
                thickness=2,
            )

            if identity:
                cv.putText(
                    img=img,
                    text=identity,
                    org=(bbox[0], bbox[1] - 10),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=1e-3 * img.shape[0],
                    color=(0, 255, 0),
                    thickness=2,
                )

        for bbox, identity in zip(bboxes, identities):
            add_rectangle_and_text(image, bbox, identity)

        if save_path:
            cv.imwrite(save_path, image)
            logging.info(f"Bounding boxes saved to {save_path}")

        if window_name is not None:
            cv.imshow(window_name, image)
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()  # esc to close the window

        if return_image:
            return image

    def detect_faces(
        self,
        image: PIL.Image.Image | np.ndarray,
        fix_bbox: bool = True,
        save_path: str | None = None,
        return_image: bool = False,
    ) -> None:
        """
        Detect the position of faces in an image using an MTCNN detector or an OpenCV Yunet detector
        and display the bounding boxes of the detected faces.

        Parameters
        ----------
        image: PIL.Image, np.ndarray
            PIL Image or numpy array representing the image.
        fix_bbox : bool, default: True
            Adjusts the dimensions of the bounding boxes so they do not exceed the
            dimensions of the image. This avoids issues when trying to represent
            bounding boxes of faces that are at the edge of the image.
        save_path: str, default None
            Path to save the output image with bounding boxes. If `None`, the image
            is displayed but not saved.
        return_image: bool, default False
            If `True`, the function returns the modified image with bounding boxes.

        Returns
        ----------
        None

        """

        if not isinstance(image, (np.ndarray, PIL.Image.Image)):
            raise Exception(
                f"`image` must be `np.ndarray`, `PIL.Image`. Got {type(image)}."
            )

        bboxes, _ = self.detect_bboxes(image, fix_bbox=fix_bbox)
        image = self.display_bounding_boxes(
            image=image, bboxes=bboxes, save_path=save_path, return_image=return_image
        )

        return image

    def extract_faces(
        self,
        image: PIL.Image.Image | np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Extract faces detected in the image using the MTCNN detector or OpenCV Yunet detector
        and return them as a tensor of images. Faces are resized to 160x160 pixels and normalized
        to the range [-1, 1] to be compatible with the InceptionResnetV1 model.

        Only faces with a probability higher than `self.min_confidence_detector` are returned.

        Parameters
        ----------
        image: PIL.Image,  np.ndarray
            PIL Image or numpy array representing the image.

        Returns
        -------
        faces: torch.Tensor
            Tensor of extracted face regions with shape
            [faces, 3, output_img_size[0], output_img_size[1]]. Values are in the range [-1, 1].
        probs: numpy array
            Numpy array with the confidence (probability) of each detected face

        """

        if not isinstance(image, (np.ndarray, PIL.Image.Image)):
            raise Exception(
                f"`image` must be np.ndarray, PIL.Image. Got {type(image)}."
            )

        if isinstance(self.detector, MTCNN):
            faces, probs = self.detector(image, return_prob=True)
        else:
            bboxes, probs = self.detect_bboxes(image, fix_bbox=True)
            faces = self._extract_faces_using_bboxes(
                image=image, bboxes=bboxes, output_img_size=[160, 160]
            )
        if len(faces) > 0:
            valid_indices = probs > self.min_confidence_detector
            faces = faces[valid_indices]
            probs = probs[valid_indices]
        else:
            faces = torch.empty((0, 3, 160, 160), dtype=torch.float32)
            probs = np.array([])

        return faces, probs

    def _extract_faces_using_bboxes(
        self,
        image: PIL.Image.Image | np.ndarray,
        bboxes: np.ndarray,
        output_img_size: list | tuple | np.ndarray = [160, 160],
    ) -> torch.Tensor:
        """
        Extract the regions of an image contained within bounding boxes and resize them
        to a specified size. The extracted regions are returned as an array of images.

        Parameters
        ----------
        image: PIL.Image,  np.ndarray
            PIL Image or numpy array representing the image.
        bboxes: np.ndarray
            Numpy array with bounding boxes of faces present in the image.
            Each bounding box is an array of 4 values defining the coordinates of
            the top-left corner and the bottom-right corner.

                 (box[0],box[1])------------
                        |                  |
                        |                  |
                        |                  |
                        ------------(box[2],box[3])
        output_img_size: list, tuple, np.ndarray, default [160, 160]
            Size of the output images in pixels.

        Returns
        -------
        faces: torch.Tensor
            Tensor of extracted face regions with shape
            [faces, 3, output_img_size[0], output_img_size[1]]. Values are in the range [-1, 1].

        """

        if not isinstance(image, (np.ndarray, PIL.Image.Image)):
            raise Exception(
                f"`image` must be np.ndarray, PIL.Image. Got {type(image)}."
            )

        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        height, width = image.shape[:2]
        bboxes = np.clip(bboxes, [0, 0, 0, 0], [width, height, width, height])

        if len(bboxes) == 0:
            faces = torch.empty(
                (0, 3, output_img_size[0], output_img_size[1]), dtype=torch.float32
            )
        else:
            faces = [
                image[int(y1) : int(y2), int(x1) : int(x2)] for x1, y1, x2, y2 in bboxes
            ]
            faces = torch.stack(
                [
                    torch.tensor(
                        cv.resize(
                            (face - 127.5) / 128.0,
                            tuple(output_img_size),
                            interpolation=cv.INTER_AREA,
                        )
                    )
                    for face in faces
                ]
            )
            faces = faces.permute(0, 3, 1, 2)
            faces = faces.float()

        return faces

    def calculate_embeddings(
        self,
        face_images: torch.Tensor | np.ndarray,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Calculate the embedding (encoding) of faces using the InceptionResnetV1 model
        from the facenet_pytorch library.

        Parameters
        ----------
        face_images: torch.Tensor, np.ndarray
            Images representing the faces with shape=[n_faces, 3, width, height].
            Values must be floats in the range [-1, 1].
        batch_size: int, default 32
            Number of faces to process in each batch.

        Returns
        -------
        embeddings: torch.Tensor
            Tensor with the embeddings of the faces, shape=[n_faces, 512].

        """
        if not isinstance(face_images, (np.ndarray, torch.Tensor)):
            raise Exception(
                f"`face_images` must be np.ndarray or torch.Tensor. Got {type(face_images)}."
            )

        if face_images.ndim != 4:
            raise Exception(
                f"`face_images` must be np.ndarray with dimensions [n_faces, width, height, 3]."
                f" Got {face_images.ndim}."
            )

        if isinstance(face_images, np.ndarray):
            face_images = torch.tensor(face_images)

        face_images = face_images.to(self.device)

        embeddings = []
        n_faces = face_images.shape[0]
        logging.info(f"Processing {n_faces} faces in batches of {batch_size}.")

        with torch.no_grad():
            for i in range(0, n_faces, batch_size):
                batch_faces = face_images[i : i + batch_size]
                batch_embeddings = self.encoder.forward(batch_faces)
                embeddings.append(batch_embeddings)

        embeddings = torch.cat(embeddings, dim=0)

        return embeddings

    def identify_faces(
        self,
        embeddings: torch.Tensor | np.ndarray
    ) -> tuple[list[str | None], list[float | None]]:
        """
        Given a set of new embeddings and a reference dictionary, calculate the similarity
        between each new embedding and the reference embeddings. If the similarity exceeds
        a specified threshold, return the identity of the person.

        Parameters
        ----------
        embeddings: torch.Tensor or np.ndarray
            Tensor or numpy array with shape [n_faces, embedding_dim] representing the embeddings
            of the faces to be identified. Values must be floats in the range [-1, 1].

        Returns
        -------
        identities: list[str | None]
            List of identities corresponding to each embedding. If the similarity
            exceeds the threshold, the identity is returned; otherwise, None is returned.
        similarity_values: list[float | None]
            List of similarity scores corresponding to each embedding's best match or None.
        """

        if self.reference_embeddings_values is None or self.reference_embeddings is None:
            raise ValueError(
                "Reference embeddings are not set. Please use `load_reference_embeddings`."
            )

        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        embeddings = embeddings.to(self.device)

        if self.similarity_metric == "euclidean":
            distances = torch.cdist(embeddings, self.reference_embeddings_values, p=2)
            similarities = 1 - distances / 2
            similarities = similarities.detach().numpy()

        if self.similarity_metric == "cosine":
            embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            reference_norm = torch.nn.functional.normalize(
                self.reference_embeddings_values, p=2, dim=1
            )
            similarities = torch.mm(embeddings_norm, reference_norm.T)
            #similarities = similarities.detach().numpy()
            similarities = similarities.detach().cpu().numpy()

        identities = []
        similarity_values = []

        for i in range(embeddings.shape[0]):
            max_index = np.argmax(similarities[i])
            max_similarity = similarities[i, max_index]

            if max_similarity >= self.similarity_threshold:
                identities.append(self.reference_embeddings_keys[max_index])
                similarity_values.append(float(max_similarity))
            else:
                identities.append(None)
                similarity_values.append(None)

        if self.verbose:
            print("----------------")
            print("Identified faces")
            print("----------------")
            for i, identity in enumerate(identities):
                if identity is not None:
                    print(
                        f"Face {i}: Identity: {identity}, Similarity: {similarity_values[i]:.2f}"
                    )
                else:
                    print(
                        f"Face {i}: No identity found, Similarity: {similarity_values[i]}"
                    )

        return identities, similarity_values

    def detect_and_identify_faces(
        self,
        image: PIL.Image.Image | np.ndarray,
        fix_bbox: bool = True,
        save_path: str | None = None,
        return_image: bool = False,
    ) -> PIL.Image.Image | None:
        """
        Detect and identify faces in an image using the MTCNN detector or OpenCV Yunet detector and
        the InceptionResnetV1 encoder.

        Parameters
        ----------
        image: PIL.Image | np.ndarray
            PIL Image or numpy array representing the image.
        fix_bbox: bool, default: True
            Adjusts the dimensions of the bounding boxes so they do not exceed the
            dimensions of the image. This avoids issues when trying to represent
            bounding boxes of faces that are at the edge of the image.
        save_path: str, default None
            Path to save the output image with bounding boxes. If `None`, the image
            is displayed but not saved.
        return_image: bool, default False
            If `True`, the function returns the modified image with bounding boxes
            and identities but does not display it.

        Returns
        -------
        image: PIL.Image.Image | None
            If `return_image` is `True`, returns the modified image with bounding boxes
            and identities. If `return_image` is `False`, returns `None`. If no faces
            are detected, the original image is returned with no modifications.

        """
        bboxes, probs = self.detect_bboxes(image=image, fix_bbox=fix_bbox)

        if len(bboxes) > 0:
            valid_indices = probs > self.min_confidence_detector
            bboxes = bboxes[valid_indices]

        if len(bboxes) == 0:
            logging.info("No faces detected in the image.")
            image = self.display_bounding_boxes(
                image=image,
                bboxes=bboxes,
                identities=None,
                save_path=save_path,
                return_image=return_image,
            )
        else:
            faces = self._extract_faces_using_bboxes(image=image, bboxes=bboxes)

            embeddings = self.calculate_embeddings(
                face_images=faces,
            )

            identities, probs_identities = self.identify_faces(embeddings=embeddings)

            image = self.display_bounding_boxes(
                image=image,
                bboxes=bboxes,
                identities=identities,
                save_path=save_path,
                return_image=return_image,
            )

        return image

    def detect_and_identify_faces_video(
        self,
        video_path: str,
        output_path: str = None,
        fix_bbox: bool = True,
        show: bool = False,
        save: bool = False,
    ) -> None:
        """
        Detect and identify faces frame-by-frame in a video.

        Parameters
        ----------
        video_path: str
            Path to the input video file.
        output_path: str, optional
            Path to save the annotated video (used if save=True).
        fix_bbox: bool
            Whether to adjust bounding boxes.
        show: bool
            If True, display the video frames in a window.
        save: bool
            If True, write the output to a video file.

        Returns
        -------
        None
        """
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        fps = int(cap.get(cv.CAP_PROP_FPS))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        if save:
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        verbose = self.verbose
        self.set_params(verbose=False)

        for frame in tqdm(
            frame_generator(cap), total=frame_count, desc="Processing video frames"
        ):

            image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            bboxes, probs = self.detect_bboxes(image=image, fix_bbox=fix_bbox)

            if len(bboxes) > 0:
                valid_indices = probs > self.min_confidence_detector
                bboxes = bboxes[valid_indices]

            if len(bboxes) == 0:
                annotated_image = self.display_bounding_boxes(
                    image=image, bboxes=bboxes, return_image=True
                )
            else:
                faces = self._extract_faces_using_bboxes(image=image, bboxes=bboxes)
                embeddings = self.calculate_embeddings(face_images=faces)
                identities, probs_identities = self.identify_faces(
                    embeddings=embeddings
                )

                annotated_image = self.display_bounding_boxes(
                    image=image, bboxes=bboxes, identities=identities, return_image=True
                )

            annotated_frame = cv.cvtColor(np.array(annotated_image), cv.COLOR_RGB2BGR)

            if show:
                cv.imshow("Face Identification", annotated_frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            if save:
                out.write(annotated_frame)

        self.set_params(verbose=verbose)

        cap.release()
        if save:
            out.release()
        if show:
            cv.destroyAllWindows()

    def detect_and_identify_faces_webcam(
        self,
        capture_index: int = 0,
        output_path: str = None,
        fix_bbox: bool = True,
        show: bool = True,
        save: bool = False,
        skip_frames: int = 0,
    ) -> None:
        """
        Detect and identify faces from the webcam stream in real time.

        Parameters
        ----------
        capture_index: int, default 0
            Index of the webcam to use (0 = default webcam).
        output_path: str, optional
            Path to save the annotated video (used if save=True).
        fix_bbox: bool
            Whether to adjust bounding boxes.
        show: bool
            If True, display the webcam frames in a window.
        save: bool
            If True, write the output to a video file.
        skip_frames: int
            Number of frames to skip between detections (0 = no skip).

        Returns
        -------
        None
        """
        cap = cv.VideoCapture(capture_index)
        if not cap.isOpened():
            raise IOError("Cannot access the webcam")

        fps = 30
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        if save:
            if output_path is None:
                raise ValueError("You must provide output_path when save=True")
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        verbose = self.verbose
        self.set_params(verbose=False)

        frame_index = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_index % skip_frames == 0:
                    image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                    bboxes, probs = self.detect_bboxes(image=image, fix_bbox=fix_bbox)

                    if len(bboxes) > 0:
                        valid_indices = probs > self.min_confidence_detector
                        bboxes = bboxes[valid_indices]

                    if len(bboxes) == 0:
                        annotated_image = self.display_bounding_boxes(
                            image=image, bboxes=bboxes, return_image=True
                        )
                    else:
                        faces = self._extract_faces_using_bboxes(image=image, bboxes=bboxes)
                        embeddings = self.calculate_embeddings(face_images=faces)
                        identities, probs_identities = self.identify_faces(
                            embeddings=embeddings
                        )

                        annotated_image = self.display_bounding_boxes(
                            image=image,
                            bboxes=bboxes,
                            identities=identities,
                            return_image=True,
                        )

                    annotated_frame = cv.cvtColor(
                        np.array(annotated_image), cv.COLOR_RGB2BGR
                    )
                else:
                    annotated_frame = frame

                if show:
                    cv.imshow("Webcam Face Identification", annotated_frame)
                    if cv.waitKey(1) & 0xFF == ord("q"):
                        break

                if save:
                    out.write(annotated_frame)

        finally:
            self.set_params(verbose=verbose)
            cap.release()
            if save:
                out.release()
            if show:
                cv.destroyAllWindows()

    def load_reference_embeddings(
        self,
        reference_embeddings: dict | ReferenceEmbeddings | str,
    ) -> None:
        """
        Load reference embeddings. If a dictionary is provided, it should be structured such that
        the key represents the identity of the person, and the value is the embedding of their face.
        If a `ReferenceEmbeddings` object is provided the reference embeddings will be set from it.
        If a file path is provided, the embeddings will be loaded from the specified file.

        Parameters
        ----------
        reference_embeddings: dict | ReferenceEmbeddings | str
            Dictionary containing reference embeddings, a `ReferenceEmbeddings` object,
            or a file path to load the embeddings from. The key represents the identity of the person,
            and the value is the embedding of their face.

        Returns
        -------
        None
        """
        if not isinstance(reference_embeddings, (dict, ReferenceEmbeddings, str)):
            raise ValueError(
                f"`reference_embeddings` must be a dictionary, a `ReferenceEmbeddings` object, "
                f"or a file path string. Got {type(reference_embeddings)}."
            )

        if isinstance(reference_embeddings, str):
            if not os.path.exists(reference_embeddings):
                raise FileNotFoundError(f"File not found: {reference_embeddings}")
            reference_embeddings = load(reference_embeddings)

        if not isinstance(reference_embeddings, (dict, ReferenceEmbeddings)):
            raise ValueError(
                f"`reference_embeddings` must be a dictionary or a `ReferenceEmbeddings` object. "
                f"Got {type(reference_embeddings)}."
            )

        if isinstance(reference_embeddings, ReferenceEmbeddings):
            reference_embeddings = reference_embeddings.reference_embeddings

        if not reference_embeddings:
            raise ValueError("`reference_embeddings` cannot be an empty dictionary.")

        reference_embeddings = reference_embeddings.copy()
        for key, value in reference_embeddings.items():
            if not isinstance(value, (np.ndarray, torch.Tensor)):
                raise ValueError(
                    f"Expected numpy array or torch tensor for embedding, "
                    f"but got {type(value)} for key '{key}'."
                )
            if isinstance(value, np.ndarray):
                reference_embeddings[key] = torch.tensor(value, dtype=torch.float32)

        self.reference_embeddings = reference_embeddings
        self.reference_embeddings_keys = list(reference_embeddings.keys())
        self.reference_embeddings_values = torch.stack(
            list(reference_embeddings.values())
        ).to(self.device)

    def update_reference_embeddings(
        self, new_reference_embeddings: dict, overwrite: bool = False
    ) -> None:
        """
        Update the reference embeddings with new embeddings. If `overwrite` is True,
        existing embeddings are replaced; otherwise, if the identity already exists,
        the new embedding is combined with the existing one using the average.

        Parameters
        ----------
        new_reference_embeddings: dict
            Dictionary containing new reference embeddings to be added or updated.
            The key represents the identity of the person, and the value is the embedding of their face.
        overwrite: bool, default False
            If True, existing embeddings are replaced with new ones. If False, and
            an identity already exists, the new embedding is combined with the existing
            one using the average.

        Returns
        -------
        None
        """

        if not isinstance(new_reference_embeddings, dict):
            raise ValueError(
                f"Expected a dictionary, but got {type(new_reference_embeddings)}."
            )

        new_reference_embeddings = new_reference_embeddings.copy()

        for key, value in new_reference_embeddings.items():
            if not isinstance(value, (np.ndarray, torch.Tensor)):
                raise ValueError(
                    f"Expected numpy array or torch tensor for embedding, "
                    f"but got {type(value)} for key '{key}'."
                )
            if isinstance(value, np.ndarray):
                new_reference_embeddings[key] = torch.tensor(value, dtype=torch.float32)

        if overwrite:
            self.reference_embeddings.update(new_reference_embeddings)
        else:
            for key, value in new_reference_embeddings.items():
                if key in self.reference_embeddings:
                    self.reference_embeddings[key] = (
                        self.reference_embeddings[key] + value
                    ) / 2
                else:
                    self.reference_embeddings[key] = value

        self.reference_embeddings_keys = list(self.reference_embeddings.keys())
        self.reference_embeddings_values = torch.stack(
            list(self.reference_embeddings.values())
        ).to(self.device)

    def set_params(self, **kwargs):
        """
        Set parameters for the face detector and encoder.

        Parameters
        ----------
        kwargs: dict
            Dictionary of parameters to set. Valid keys are:
            - `min_face_size`: int, minimum size of faces to be detected.
            - `thresholds`: list[float], detection thresholds for MTCNN.
            - `min_confidence_detector`: float, minimum confidence for detected faces.
            - `similarity_threshold`: float, minimum similarity for identity assignment.
            - `similarity_metric`: str, metric used for similarity calculation ('euclidean' or 'cosine').
            - `verbose`: bool, whether to display process information.
            - `device`: str, device where the models are executed.
            - `detector`: facenet_pytorch.models.mtcnn.MTCNN, MTCNN model for face detection.
            - `encoder`: facenet_pytorch.models.inception_resnet_v1.InceptionResnetV1, InceptionResnetV1 model for face embeddings.

        Returns
        -------
        None
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")


class ReferenceEmbeddings:
    """
    Class to manage reference embeddings for face recognition tasks using the facenet_pytorch library.

    This class allows loading, updating, and managing reference embeddings
    used for identifying faces in images. It provides methods to load embeddings
    from a file, update them with new embeddings, and retrieve the current embeddings.

    Parameters
    ----------
    folder_path: str
        Path to the directory containing reference images. The expected structure
        in this directory is:
            - One folder per identity. The folder name is used as a unique identifier.
            - Each folder may contain one or more images of the individual.
              If there are multiple images, the average embedding of all images
              is calculated. Reference images should only contain the face of
              the individual.
    min_face_size: int, default: 20
        Minimum size of faces to be detected by the MTCNN network.
    thresholds: list[float], default: [0.6, 0.7, 0.7]
        Detection thresholds for each of the three networks forming the MTCNN detector.
    min_confidence_detector: float, default: 0.5
        Minimum confidence (probability) required for a detected face to be included
        in the results.
    device: str | None, default: None
        Device where the models are executed. If None, it defaults to 'cuda:0' if a GPU is available, otherwise 'cpu'.
    verbose: bool, default: True
        Whether to display process information during execution. If True, additional
        information about the detection and identification process will be printed to the console.

    Attributes
    ----------
    folder_path: str
        Path to the directory containing reference images.
    device: str
        Device where the models are executed.
    min_face_size: int
        Minimum size of faces to be detected by the MTCNN network.
    thresholds: list[float]
        Detection thresholds for each of the three networks forming the MTCNN detector.
    min_confidence_detector: float
        Minimum confidence (probability) required for a detected face to be included
        in the results.
    reference_embeddings: dict
        Dictionary containing reference embeddings for individuals. The key is the identity
        of a person, and the value is the embedding of their face that can be used for face recognition tasks.
    identities: list[str]
        List of identities corresponding to the reference embeddings.
    n_images_per_identity: dict
        Dictionary containing the number of images per identity in the reference embeddings.
        The key is the identity of a person, and the value is the number of images associated
        with that identity in the reference embeddings.
    verbose: bool
        Whether to display process information during execution.

    """

    def __init__(
        self,
        folder_path: str,
        save_path: str | None = None,
        min_face_size: int = 20,
        thresholds: list[float] = [0.6, 0.7, 0.7],
        min_confidence_detector: float = 0.5,
        device: str | None = None,
        verbose: bool = True,
    ):

        if not os.path.isdir(folder_path):
            raise Exception(f"Directory {folder_path} does not exist.")

        if len(os.listdir(folder_path)) == 0:
            raise Exception(f"Directory {folder_path} is empty.")

        logging.info("Initializing FaceRecognizer")
        face_detector = FaceRecognizer(
            keep_all=True,
            min_face_size=min_face_size,
            thresholds=thresholds,
            min_confidence_detector=min_confidence_detector,
            device=device,
        )

        self.folder_path = folder_path
        self.save_path = save_path
        self.face_detector = face_detector
        self.device = device
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.min_confidence_detector = min_confidence_detector
        self.verbose = verbose
        self.reference_embeddings = {}
        self.identities = []
        self.n_images_per_identity = {}

    def __repr__(self):
        text = (
            f"-------------------\n",
            f"ReferenceEmbeddings\n",
            f"-------------------\n",
            f"Number of identities: {len(self.identities)}\n",
            f"Number of images per identity: {self.n_images_per_identity}\n",
            f"Source folder: {self.folder_path}\n",
            f"Save path: {self.save_path}\n",
            f"Device: {self.device}\n",
            f"Minimum face size: {self.min_face_size}\n",
            f"Detection thresholds: {self.thresholds}\n",
            f"Minimum confidence for detection: {self.min_confidence_detector}\n",
            f"Verbose: {self.verbose}",
        )
        return "".join(text)

    def calculate_reference_embeddings(self) -> None:
        """
        Calculate reference embeddings for individuals in the specified folder.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        folders = glob.glob(self.folder_path + "/*")

        for folder in folders:
            identity = folder.split(os.sep)[-1]

            if self.verbose:
                print(f"Processing identity: {identity}")
            embeddings = []
            image_paths = glob.glob(folder + "/*.jpg")
            image_paths.extend(glob.glob(folder + "/*.jpeg"))
            image_paths.extend(glob.glob(folder + "/*.tif"))
            image_paths.extend(glob.glob(folder + "/*.png"))
            logging.info(f"Total reference images: {len(image_paths)}")

            for image_path in image_paths:
                if self.verbose:
                    print(f"  Reading image: {image_path}")
                image = Image.open(image_path).convert("RGB")
                faces, probs = self.face_detector.extract_faces(image=image)

                if len(faces) == 0:
                    if self.verbose:
                        print(f"  No faces detected in image: {image_path}.")
                    continue

                if len(faces) > 1:
                    if self.verbose:
                        print(
                            f"  More than 2 faces detected in image, "
                            f"The face with the highest confidence will be used: "
                            f"{image_path}"
                        )
                    max_index = np.argmax(probs)
                    faces = faces[max_index : max_index + 1]

                embedding = self.face_detector.calculate_embeddings(
                    face_images=faces,
                )
                embeddings.append(embedding.squeeze(0))

            #average_embedding = np.array(embeddings).mean(axis=0)
            embeddings_on_cpu = [e.cpu() for e in embeddings]
            average_embedding = np.array(embeddings_on_cpu).mean(axis=0)
            self.reference_embeddings[identity] = average_embedding
            self.identities.append(identity)
            self.n_images_per_identity[identity] = len(embeddings)

        if self.save_path:
            print(f"Saving Reference Embeddings to {self.save_path}")
            dump(self.reference_embeddings, self.save_path)
