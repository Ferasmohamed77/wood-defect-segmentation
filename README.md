# Wood Defect Segmentation System

## Overview
This project is an AI-based image segmentation system designed to detect defects in wood surfaces. The model identifies and highlights defects such as small holes, large holes, and scratches.

## Features
- Pixel-level defect segmentation
- Detects multiple defect types:
  - Small holes
  - Large holes
  - Scratches
- Real-time inference using camera
- Visual highlighting of detected defects

## Tech Stack
- Python
- OpenCV
- Deep Learning (Segmentation Model)

## How It Works
Camera/Image → Preprocessing → Segmentation Model → Mask Output → Highlight Defects

## Demo

### Example Results
(Add your images here)

## Dataset Sample
A small subset of wood images is included in the `dataset_sample` folder.

## Installation
pip install -r requirements.txt

## Run
python src/segmentation_inference.py

## Use Case
- Industrial quality inspection
- Automated defect detection
- Smart manufacturing systems