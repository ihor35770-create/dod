"""
Object Detection Engine using SAHI for small object detection
"""

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from typing import List, Tuple, Any


class ObjectDetection:
    """
    Object detection class using SAHI (Slicing Aided Hyper Inference)
    for improved small object detection in large images.
    """

    def __init__(self, model_path: str, model_type: str = "yolov8", confidence_threshold: float = 0.25):
        """
        Initialize the object detection model.

        Args:
            model_path: Path to the model file (e.g., "models/yolo11m.pt")
            model_type: Type of model ("yolov8", "yolov5", "detectron2", etc.)
            confidence_threshold: Minimum confidence score for detections
        """
        self.model_path = model_path
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold

        # Initialize SAHI detection model
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device="cpu"  # Use CPU by default, can be changed to "cuda" if GPU available
        )

    def detect(self, image: np.ndarray, slice_height: int = 512, slice_width: int = 512,
               overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2) -> Tuple[List[List[int]], List[str], List[float]]:
        """
        Detect objects in the given image using SAHI slicing.

        Args:
            image: Input image as numpy array (BGR format)
            slice_height: Height of each slice
            slice_width: Width of each slice
            overlap_height_ratio: Overlap ratio between slices (height)
            overlap_width_ratio: Overlap ratio between slices (width)

        Returns:
            Tuple of (bboxes, labels, confidences)
            - bboxes: List of [x1, y1, x2, y2] coordinates
            - labels: List of class labels
            - confidences: List of confidence scores
        """
        # Convert BGR to RGB for SAHI
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Perform sliced prediction
        result = get_sliced_prediction(
            image=image_rgb,
            detection_model=self.detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio
        )

        # Extract results
        bboxes = []
        labels = []
        confidences = []

        for prediction in result.object_prediction_list:
            # Get bounding box coordinates
            bbox = prediction.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)

            # Get class label and confidence
            label = prediction.category.name
            confidence = prediction.score.value

            # Only keep detections above confidence threshold
            if confidence >= self.confidence_threshold:
                bboxes.append([x1, y1, x2, y2])
                labels.append(label)
                confidences.append(confidence)

        return bboxes, labels, confidences

    def detect_single_slice(self, image: np.ndarray) -> Tuple[List[List[int]], List[str], List[float]]:
        """
        Detect objects without slicing (for comparison or small images).

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Tuple of (bboxes, labels, confidences)
        """
        # Convert BGR to RGB for SAHI
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Perform prediction without slicing
        result = get_sliced_prediction(
            image=image_rgb,
            detection_model=self.detection_model,
            slice_height=image.shape[0],  # Use full image height
            slice_width=image.shape[1],   # Use full image width
            overlap_height_ratio=0.0,     # No overlap
            overlap_width_ratio=0.0
        )

        # Extract results
        bboxes = []
        labels = []
        confidences = []

        for prediction in result.object_prediction_list:
            bbox = prediction.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            label = prediction.category.name
            confidence = prediction.score.value

            if confidence >= self.confidence_threshold:
                bboxes.append([x1, y1, x2, y2])
                labels.append(label)
                confidences.append(confidence)

        return bboxes, labels, confidences