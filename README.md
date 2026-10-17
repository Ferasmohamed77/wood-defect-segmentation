# Wood Defect Segmentation System

A real-time visual inspection system that finds and outlines defects on wood surfaces. A YOLOv8n-seg model, exported to ONNX, runs on a Raspberry Pi with either a USB webcam or a Pi camera and segments three defect types at pixel level: **big holes, small holes and scratches**. Built during my AI Engineering internship at Control Easy Technology (2026).

![Inspection GUI showing segmented defects](demo/WhatsApp%20Image%202026-04-29%20at%2011.40.18%20AM.jpeg)

*Inspection GUI on a live 640×480 frame: 14 defects found (8 scratches, 3 big holes, 3 small holes), with the per-frame processing time and average confidence shown on the right.*

## How It Works

```
Camera frame → letterbox to 640×640 → YOLOv8n-seg (ONNX Runtime, CPU)
            → confidence filter + NMS → mask decoding → grain filter → overlay + counts
```

1. **Capture.** Frames come from a USB webcam (V4L2) or the Pi camera, with exposure and gain adjustable in the GUI.
2. **Inference.** The ONNX model returns boxes, class scores and mask coefficients; masks are decoded from the prototype output and cropped to each box.
3. **Post-processing.** Non-maximum suppression removes duplicates, and an optional wood-grain filter rejects detections that look like natural grain rather than defects.
4. **Display.** Defects are overlaid in a per-class colour, and the GUI shows counts per class, pass/fail status and average confidence.

## Model

| Item | Detail |
|---|---|
| Architecture | YOLOv8n-seg (Ultralytics), exported to ONNX |
| Classes | Big Hole, Scratch, Small Hole |
| Input size | 640 × 640 |
| Dataset | Wood images captured with the tools in `src/*_Dataset.py`, labelled in Roboflow |
| Training | Google Colab |
| Models | `models/best_usb.onnx` (USB webcam), `models/best.onnx` (Pi camera) |

## Project Structure

```
wood-defect-segmentation/
├── src/
│   ├── USB_Webcam_System_Segmentation.py   # Live inspection GUI, USB webcam
│   ├── Pi_Camera_System_Segmentation.py    # Live inspection GUI, Pi camera
│   ├── USB_Webcam_Dataset.py               # Dataset capture tool, USB webcam
│   └── Pi_Camera_Dataset.py                # Dataset capture tool, Pi camera
├── models/          # Trained ONNX models
├── dataset_sample/  # Sample training images
├── demo/            # Screenshots
└── docs/            # How it works, setup and system notes
```

## Dataset Capture Tool

![Dataset capture tool](demo/WhatsApp%20Image%202026-04-29%20at%2011.33.12%20AM.jpeg)

Images are captured into train / val folders by class, with exposure and gain control, before labelling in Roboflow.

## Installation

Runs on a Raspberry Pi (the scripts use `RPi.GPIO`).

```bash
pip install -r requirements.txt
```

## Run

Set `Model_Path` in the `Config` class at the top of the script to the model file in `models/`, then run:

```bash
# USB webcam
python3 src/USB_Webcam_System_Segmentation.py

# Pi camera
python3 src/Pi_Camera_System_Segmentation.py
```

## Use Cases

Industrial quality inspection, automated defect detection on production lines, and grading of timber surfaces.
