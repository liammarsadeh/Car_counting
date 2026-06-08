# 🚗 Real-Time Vehicle Counter using YOLOv8 and SORT

A computer vision project that performs real-time vehicle detection, tracking, and counting from video footage using **YOLOv8** for object detection and **SORT (Simple Online and Realtime Tracking)** for multi-object tracking.

## Overview

This project detects vehicles from a traffic video, tracks each vehicle with a unique ID, and counts vehicles as they cross a predefined counting line.

The system combines:

- **YOLOv8** → Vehicle Detection
- **SORT** → Object Tracking
- **OpenCV** → Video Processing
- **CVZone** → Visualization Utilities

---

## Features

- Real-time vehicle detection
- Vehicle tracking with unique IDs
- Counts vehicles only once
- Supports cars and trucks
- Region-of-interest masking
- Visual tracking boxes and centroids
- Crossing-line counting mechanism

---

## Project Structure

```text
Project-1-Car-Counter/
│
├── YOLO_CAR_COUNT.py        # Main vehicle counting script
├── sort.py                 # SORT tracking algorithm
├── yolov8n.pt              # YOLOv8 pretrained model
├── car.mp4                 # Input traffic video
├── car_edited.png          # ROI mask image
├── mouse_event.ipynb       # Utility notebook
├── YOLO_with_WEBCAM.ipynb  # Webcam detection notebook
└── __pycache__/
