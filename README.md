# Drone object detection

Inspired by Murtaza (commercial courses)
https://www.computervision.zone/lessons/code-and-files-3/

To get som ideas on SAHI look at https://github.com/obss/sahi/blob/main/README.md


## project structure

(download the models like yolo*.pt and a data file *.mp4)

├── 14691540-uhd_3840_2160_24fps.mp4
├── coco.names
├── drone_object_detection.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── engine
│   ├── __init__.py
│   ├── object_detection.py
│   └── __pycache__
│       ├── __init__.cpython-312.pyc
│       └── object_detection.cpython-312.pyc
├── frozen_inference_graph.pb
├── main.py
├── models
│   └── yolo11m.pt
├── pyproject.toml
├── README.md
├── requirements.txt
├── sod
│   └── small.py
├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
├── test_detection.py
└── yolov8m.pt

6 directories, 21 files

## Requirements

See requirements.txt

