#!/usr/bin/env python3
"""
Test script for the ObjectDetection engine
"""

import cv2
import numpy as np
from engine.object_detection import ObjectDetection

def test_object_detection():
    """Test the ObjectDetection class with a simple image"""

    # Create a simple test image (black background with white rectangle)
    test_image = np.zeros((640, 480, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)

    # Initialize object detector
    print("Initializing ObjectDetection...")
    od = ObjectDetection("models/yolo11m.pt")

    # Test detection
    print("Running detection...")
    bboxes, labels, confs = od.detect(test_image)

    print(f"Found {len(bboxes)} objects")
    for i, (bbox, label, conf) in enumerate(zip(bboxes, labels, confs)):
        print(f"  Object {i+1}: {label} at {bbox} with confidence {conf:.2f}")

    print("Test completed successfully!")

if __name__ == "__main__":
    test_object_detection()