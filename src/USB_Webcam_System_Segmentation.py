import sys
import os
import time
import logging
import threading
import subprocess
import re

from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage

import cv2
import numpy as np
import onnxruntime as ort
import RPi.GPIO as GPIO


# ---------------- CONFIG ----------------
class Config:
    # ===== Model =====
    Model_Path = r"/home/controleasy/controleasyvisionproject/best_usb.onnx"
    Model_ImgSz = 640

    # Detection thresholds
    Conf_Thres = 0.12
    IoU_Thres = 0.45
    Mask_Thres = 0.40

    # MUST match training order exactly
    # 0 = Big Hole, 1 = Scratches, 2 = Small Hole
    Classes = ["Big Hole", "Scratches", "Small Hole"]

    # ===== Grain / texture filter =====
    USE_GRAIN_FILTER = True
    GRAIN_MinAnomalyMean = 0.08
    GRAIN_MinArea = 120
    GRAIN_MaxAreaRatio = 0.35

    HOLE_MaxAspect = 4.5
    HOLE_MinSolidity = 0.12
    SCRATCH_MinAspect = 1.0

    # ===== GUI =====
    font = QFont("Times New Roman", 15)
    Logo_Path = r"/home/controleasy/controleasyvisionproject/Images/ControlEasy"

    # ===== Camera =====
    Video_Index = 0
    Default_Size = (640, 480)

    Exposure_Time = 250
    Min_Exposure = 1
    Max_Exposure = 10000

    Analogue_Gain = 8
    Min_Analogue = 0
    Max_Analogue = 255

    Res_Num = 0
    Fmt_Num = 0

    # ===== Timing =====
    alpha = 0.05
    Time_Wait = False
    Button_Wait = False

    # ===== GPIO =====
    PIN = 4
    Sensor_Wait = False
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


# ---------------- UTILS ----------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def xywh_to_xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_numpy(boxes, scores, iou_thres):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


def letterbox(image, new_shape=640, color=(114, 114, 114)):
    h, w = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return padded, r, (left, top)


def scale_boxes_to_original(boxes_xyxy, r, pad, orig_shape):
    left, top = pad
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes /= r

    h, w = orig_shape[:2]
    boxes[:, 0] = boxes[:, 0].clip(0, w)
    boxes[:, 2] = boxes[:, 2].clip(0, w)
    boxes[:, 1] = boxes[:, 1].clip(0, h)
    boxes[:, 3] = boxes[:, 3].clip(0, h)
    return boxes


def overlay_mask(image_bgr, mask, color, alpha=0.35):
    overlay = image_bgr.copy()
    overlay[mask] = (
        overlay[mask] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)
    return overlay


def class_color(cls_id):
    palette = [
        (0, 255, 0),      # Big Hole -> green
        (255, 0, 0),      # Scratches -> blue
        (0, 165, 255),    # Small Hole -> orange
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
    ]
    return palette[cls_id % len(palette)]


# ---------------- MASK CLEANUP ----------------
def crop_mask_to_box(mask_float, box_xyxy):
    h, w = mask_float.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(int)

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    cropped = np.zeros_like(mask_float, dtype=mask_float.dtype)
    if x2 > x1 and y2 > y1:
        cropped[y1:y2, x1:x2] = mask_float[y1:y2, x1:x2]
    return cropped


def clean_binary_mask(mask_bool):
    mask_u8 = (mask_bool.astype(np.uint8) * 255)

    kernel_open = np.ones((1, 1), np.uint8)
    kernel_close = np.ones((2, 2), np.uint8)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_open)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)

    return mask_u8 > 0


# ---------------- GRAIN FILTER ----------------
def make_anomaly_map(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=7, sigmaY=7)
    hi = cv2.absdiff(gray, blur)
    hi = cv2.normalize(hi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hi = cv2.GaussianBlur(hi, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return hi.astype(np.float32) / 255.0


def mask_stats(mask01: np.ndarray):
    area = int(mask01.sum())
    if area == 0:
        return {"area": 0, "aspect": 0.0, "solidity": 0.0}

    ys, xs = np.where(mask01 > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = int(x1 - x0 + 1)
    h = int(y1 - y0 + 1)
    aspect = max(w, h) / max(1, min(w, h))

    cnts, _ = cv2.findContours(
        (mask01 * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return {"area": area, "aspect": float(aspect), "solidity": 0.0}

    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    hull_area = max(1.0, cv2.contourArea(hull))
    sol = float(cv2.contourArea(c) / hull_area)

    return {"area": area, "aspect": float(aspect), "solidity": sol}


def keep_detection(cls_id: int, conf: float, mask_bool: np.ndarray, anomaly_map: np.ndarray) -> bool:
    if conf < Config.Conf_Thres:
        return False

    h, w = mask_bool.shape[:2]
    mask01 = mask_bool.astype(np.uint8)

    area = int(mask01.sum())
    if area == 0:
        return False

    vals = anomaly_map[mask_bool]
    if vals.size == 0:
        return False

    an_mean = float(vals.mean())
    st = mask_stats(mask01)

    if cls_id == 0:  # Big Hole
        if area < 120:
            return False
        if area / float(h * w) > Config.GRAIN_MaxAreaRatio:
            return False
        if an_mean < 0.07:
            return False
        if st["aspect"] > 5.0:
            return False
        if st["solidity"] < 0.08:
            return False
        return True

    if cls_id == 1:  # Scratches
        if area < 45:
            return False
        if an_mean < 0.035:
            return False
        if st["aspect"] < 1.0:
            return False
        return True

    if cls_id == 2:  # Small Hole
        if area < 20:
            return False
        if an_mean < 0.04:
            return False
        if st["aspect"] > 6.0:
            return False
        return True

    return True


# ---------------- MODEL ----------------
class SegModel:
    def __init__(self):
        if not os.path.exists(Config.Model_Path):
            raise FileNotFoundError(f"Model not found: {Config.Model_Path}")

        self.session = ort.InferenceSession(
            Config.Model_Path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.nc = len(Config.Classes)

        print("Loaded model:", Config.Model_Path)
        print("Model input name:", self.input_name)
        print("Classes:", Config.Classes)

    def preprocess(self, frame_bgr):
        img, r, pad = letterbox(frame_bgr, Config.Model_ImgSz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, r, pad

    def infer(self, frame_bgr):
        inp, r, pad = self.preprocess(frame_bgr)
        outs = self.session.run(None, {self.input_name: inp})

        proto = None
        pred = None

        for o in outs:
            if isinstance(o, np.ndarray) and o.ndim == 4:
                proto = o
            elif isinstance(o, np.ndarray) and o.ndim == 3:
                pred = o

        if pred is None:
            raise RuntimeError("Could not find prediction output in ONNX outputs.")
        if proto is None:
            raise RuntimeError("Could not find proto(mask) output in ONNX outputs.")

        pred = np.squeeze(pred, axis=0)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        D = pred.shape[1]
        nm = D - (4 + self.nc)
        if nm <= 0:
            raise RuntimeError(f"Bad output dims. Got D={D}, expected 4+nc+nm.")

        boxes_xywh = pred[:, 0:4]
        cls_scores = pred[:, 4:4 + self.nc]
        mask_coeff = pred[:, 4 + self.nc:]

        cls_id = np.argmax(cls_scores, axis=1)
        conf = cls_scores[np.arange(cls_scores.shape[0]), cls_id]

        keep = conf > Config.Conf_Thres
        boxes_xywh = boxes_xywh[keep]
        cls_id = cls_id[keep]
        conf = conf[keep]
        mask_coeff = mask_coeff[keep]

        if boxes_xywh.shape[0] == 0:
            return [], [], [], None, r, pad

        boxes_xyxy = xywh_to_xyxy(boxes_xywh)

        keep_idx = nms_numpy(boxes_xyxy, conf, Config.IoU_Thres)
        boxes_xyxy = boxes_xyxy[keep_idx]
        cls_id = cls_id[keep_idx]
        conf = conf[keep_idx]
        mask_coeff = mask_coeff[keep_idx]

        proto = np.squeeze(proto, axis=0)
        nm2, mh, mw = proto.shape
        if nm2 != mask_coeff.shape[1]:
            raise RuntimeError(f"Mask dim mismatch: proto nm={nm2} vs coeff nm={mask_coeff.shape[1]}")

        proto_flat = proto.reshape(nm2, -1)
        masks = sigmoid(mask_coeff @ proto_flat)
        masks = masks.reshape(-1, mh, mw)

        return boxes_xyxy, cls_id, conf, masks, r, pad

    def draw(self, frame_bgr, boxes_xyxy, cls_id, conf, masks, r, pad):
        out = frame_bgr.copy()
        h0, w0 = frame_bgr.shape[:2]

        anomaly = make_anomaly_map(frame_bgr) if Config.USE_GRAIN_FILTER else None

        kept_boxes = []
        kept_cls = []
        kept_conf = []

        if len(boxes_xyxy) > 0:
            boxes_orig = scale_boxes_to_original(np.array(boxes_xyxy), r, pad, frame_bgr.shape)
        else:
            boxes_orig = np.zeros((0, 4), dtype=np.float32)

        if masks is not None and len(masks) > 0:
            for i in range(len(masks)):
                m = masks[i]
                m640 = cv2.resize(
                    m,
                    (Config.Model_ImgSz, Config.Model_ImgSz),
                    interpolation=cv2.INTER_LINEAR
                )

                box_model = np.array(boxes_xyxy[i], dtype=np.float32)
                m640 = crop_mask_to_box(m640, box_model)

                left, top = pad
                y1p = top
                y2p = Config.Model_ImgSz - top
                x1p = left
                x2p = Config.Model_ImgSz - left
                m_unpad = m640[y1p:y2p, x1p:x2p]

                m_orig = cv2.resize(m_unpad, (w0, h0), interpolation=cv2.INTER_LINEAR)
                mask_bool = m_orig > Config.Mask_Thres

                x1o, y1o, x2o, y2o = boxes_orig[i].astype(int)
                x1o = max(0, min(x1o, w0 - 1))
                x2o = max(0, min(x2o, w0))
                y1o = max(0, min(y1o, h0 - 1))
                y2o = max(0, min(y2o, h0))

                cropped_final = np.zeros_like(mask_bool, dtype=bool)
                if x2o > x1o and y2o > y1o:
                    cropped_final[y1o:y2o, x1o:x2o] = mask_bool[y1o:y2o, x1o:x2o]
                mask_bool = cropped_final

                mask_bool = clean_binary_mask(mask_bool)
                if mask_bool.sum() == 0:
                    continue

                c = int(cls_id[i])
                cf = float(conf[i])

                if Config.USE_GRAIN_FILTER and anomaly is not None:
                    if not keep_detection(c, cf, mask_bool, anomaly):
                        continue

                kept_boxes.append(boxes_orig[i])
                kept_cls.append(c)
                kept_conf.append(cf)

                color = class_color(c)
                out = overlay_mask(out, mask_bool, color, alpha=0.35)

        counts = {}
        for i in range(len(kept_boxes)):
            x1, y1, x2, y2 = np.array(kept_boxes[i]).astype(int)
            c = int(kept_cls[i])
            cf = float(kept_conf[i])

            name = Config.Classes[c] if c < len(Config.Classes) else str(c)
            label = f"{name} {cf:.2f}"
            color = class_color(c)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                label,
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

            counts[name] = counts.get(name, 0) + 1

        avg_conf = float(np.mean(kept_conf)) if kept_conf else 0.0
        return out, counts, avg_conf


# ---------------- GUI ----------------
class GUI(QMainWindow):
    Model_Done = pyqtSignal(object, object, float)

    def __init__(self):
        super().__init__()
        self.EXP_T = None
        self.EXP_T_2 = None

        self.model = SegModel()
        self.last_avg_conf = 0.0

        Widget = QWidget()
        self.setCentralWidget(Widget)
        Widget.setFont(Config.font)

        self.Model_Done.connect(self.Update_Result)

        self.Camera_Input = self.Make_Label(None, "Panel", 1)
        self.Camera_Input.setFixedSize(Config.Default_Size[0], Config.Default_Size[1])

        self.Input_Label = self.Make_Label("<b>Camera Input", "Sunken", 5)
        self.Input_Label.setFixedHeight(40)

        self.Camera_Resolution = self.Make_Label("<b><u>Resolution</u><br>N/A", "Raised", 5)
        self.Camera_Resolution.setFixedHeight(80)

        self.Camera_Capturing_Time = self.Make_Label("<b><u>Capturing Time (seconds)</u><br>N/A", "Raised", 5)
        self.Camera_Capturing_Time.setFixedHeight(80)

        self.System_Processing_Time = self.Make_Label("<b><u>Processing Time (seconds)</u><br>N/A", "Raised", 5)
        self.System_Processing_Time.setFixedHeight(80)

        self.Class_Label = self.Make_Label("<b><u>Detections</u><br>N/A", "Raised", 5)
        self.Class_Label.setFixedHeight(80)

        self.Score = self.Make_Label("<b><u>Status</u><br>Ready", "Raised", 5)
        self.Score.setFixedHeight(80)

        self.Avg_Conf_Label = self.Make_Label("<b><u>Average Confidence</u><br>N/A", "Raised", 5)
        self.Avg_Conf_Label.setFixedHeight(80)

        self.Avg_Conf_Button = QPushButton("Show Average Confidence")
        self.Avg_Conf_Button.clicked.connect(self.Show_Avg_Confidence)

        Exit_Button = QPushButton("Press ESC To Quit")
        Exit_Button.clicked.connect(self.Closing)

        EZLogo = QLabel()
        if os.path.exists(Config.Logo_Path):
            EZLogo.setPixmap(QPixmap(Config.Logo_Path))
        EZLogo.setScaledContents(True)
        EZLogo.setMaximumSize(300, 100)

        EZLogo2 = QLabel("Machine Vision & Control Automation")
        Screen_Resolution = QApplication.primaryScreen()	#Get the (x,y) of the monitor/screen
        Screen_Width = Screen_Resolution.size().width()		#Get the width
        Font_Factor = Screen_Width  /1920					#Divide current width with base width(1920)
        Text_Size = int(60 * Font_Factor)					#Readjust the font size based on the factor
        EZLogo2.setStyleSheet(f"""
        color: white;
        background-color: black;
        border: 10px outset gray; /*Raised effect */ 
        font-size: {Text_Size}px;
        font-family: Times New Roman;   
        """)									#Setup the option
        EZLogo2.setAlignment(Qt.AlignCenter)

        EZLogo3 = QLabel()
        if os.path.exists(Config.Logo_Path):
            EZLogo3.setPixmap(QPixmap(Config.Logo_Path))
        EZLogo3.setScaledContents(True)
        EZLogo3.setMaximumSize(300, 100)

        self.Exposure_Slider = self.Make_Slider(
            Config.Min_Exposure, Config.Max_Exposure, Config.Exposure_Time, 1000
        )
        self.Exposure_Slider.valueChanged.connect(self.Exposure_Changes)

        self.Exposure_Label = self.Make_Label(
            f"<b>Exposure Time: {Config.Exposure_Time}", "Sunken", 5
        )
        self.Exposure_Edit = QLineEdit()
        self.Exposure_Edit.setPlaceholderText(
            f"Enter exposure ({self.Exposure_Slider.minimum()} to {self.Exposure_Slider.maximum()})"
        )
        self.Exposure_Edit.returnPressed.connect(self.Manual_Exposure_Changes)

        self.Analogue_Slider = self.Make_Slider(
            Config.Min_Analogue, Config.Max_Analogue, Config.Analogue_Gain, 1
        )
        self.Analogue_Slider.valueChanged.connect(self.Analogue_Changes)

        self.Analogue_Label = self.Make_Label(
            f"<b>Analogue Gain: {Config.Analogue_Gain}", "Sunken", 5
        )

        Row1 = QHBoxLayout()
        Row1.addWidget(EZLogo)
        Row1.addSpacing(50)
        Row1.addWidget(EZLogo2)
        Row1.addSpacing(50)
        Row1.addWidget(EZLogo3)

        Left = QVBoxLayout()
        Left.addWidget(self.Input_Label)
        Left.addSpacing(10)
        Left.addWidget(self.Camera_Input)
        Left.addStretch()

        Right = QVBoxLayout()
        Right.addWidget(self.Camera_Resolution)
        Right.addWidget(self.Camera_Capturing_Time)
        Right.addWidget(self.System_Processing_Time)
        Right.addWidget(self.Class_Label)
        Right.addWidget(self.Score)
        Right.addWidget(self.Avg_Conf_Label)
        Right.addWidget(self.Avg_Conf_Button)
        Right.addStretch()
        Right.addWidget(self.Exposure_Slider)
        Right.addWidget(self.Exposure_Label)
        Right.addWidget(self.Exposure_Edit)
        Right.addStretch()
        Right.addWidget(self.Analogue_Slider)
        Right.addWidget(self.Analogue_Label)
        Right.addStretch()
        Right.addWidget(Exit_Button)

        Row2 = QHBoxLayout()
        Row2.addLayout(Left)
        Row2.addSpacing(100)
        Row2.addLayout(Right)

        Main = QVBoxLayout()
        Main.setContentsMargins(20, 10, 20, 20)
        Main.addLayout(Row1)
        Main.addSpacing(20)
        Main.addLayout(Row2)
        Widget.setLayout(Main)

        self.last_annotated = None
        self.last_counts = {}

        ok = self.Camera()
        self.Timer = QTimer()
        if ok:
            self.Timer.timeout.connect(self.Update_Camera)
            self.Timer.start(30)
        else:
            logging.warning("Camera init failed; timer not started.")

    def Make_Label(self, Text, Style, Line):
        label = QLabel(Text)
        if Style == "Sunken":
            label.setFrameStyle(QLabel.Panel | QLabel.Sunken)
        elif Style == "Raised":
            label.setFrameStyle(QLabel.Panel | QLabel.Raised)
        elif Style == "Panel":
            label.setFrameStyle(QLabel.Panel)
        label.setLineWidth(Line)
        label.setAlignment(Qt.AlignCenter)
        return label

    def Make_Slider(self, minv, maxv, initial, gap):
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(minv, maxv)
        slider.setValue(initial)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(gap)
        return slider

    def Show_Avg_Confidence(self):
        if self.last_avg_conf > 0:
            self.Avg_Conf_Label.setText(
                f"<b><u>Average Confidence</u><br>{self.last_avg_conf * 100:.2f}%"
            )
        else:
            self.Avg_Conf_Label.setText("<b><u>Average Confidence</u><br>N/A")

    def Camera(self):
        dev = f"/dev/video{Config.Video_Index}"
        if not os.path.exists(dev):
            logging.warning(f"USB Webcam is not connected at {dev}")
            return False

        self.USB_Cam = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.USB_Cam.isOpened():
            logging.warning("USB Webcam cannot be accessed (busy or permissions).")
            return False

        self.Camera_List = self.Resolution_List()
        self.Keys = list(self.Camera_List.keys()) if self.Camera_List else []
        if not self.Keys:
            logging.warning("Could not read camera formats/resolutions. Using defaults.")
            self.Keys = ["DEFAULT"]
            self.Camera_List = {"DEFAULT": [f"{Config.Default_Size[0]}x{Config.Default_Size[1]}"]}

        self.Resolution_Changes()
        self.Format_Changes()
        return True

    def Update_Camera(self):
        if not hasattr(self, "USB_Cam") or self.USB_Cam is None or not self.USB_Cam.isOpened():
            return

        self.USB_Cam.set(cv2.CAP_PROP_EXPOSURE, float(Config.Exposure_Time))
        self.USB_Cam.set(cv2.CAP_PROP_GAIN, float(Config.Analogue_Gain))

        try:
            fmt = self.Keys[Config.Fmt_Num] if self.Keys else "N/A"
            self.Camera_Resolution.setText(
                f"<b><u>Resolution</u><br>{fmt}: {self.Res_Width} x {self.Res_Height} [{Config.Res_Num}]"
            )
        except Exception:
            self.Camera_Resolution.setText("<b><u>Resolution</u><br>N/A")

        self.Camera_Start_Time = time.time()
        ret, frame = self.USB_Cam.read()
        if not ret or frame is None:
            return

        self.Frame = frame.copy()

        # Stay on last result until next trigger
        show_frame = self.last_annotated if self.last_annotated is not None else self.Frame
        self.show_on_label(show_frame)

        if GPIO.input(Config.PIN) == GPIO.HIGH and not Config.Sensor_Wait:
            Config.Sensor_Wait = True
            if not Config.Time_Wait:
                threading.Thread(target=self.Trigger_Model, daemon=True).start()

        if GPIO.input(Config.PIN) != GPIO.HIGH:
            Config.Sensor_Wait = False

    def show_on_label(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        scaled = qimg.scaled(self.Camera_Input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.Camera_Input.setPixmap(QPixmap.fromImage(scaled))

    def Trigger_Model(self):
        if Config.Time_Wait:
            return
        Config.Time_Wait = True

        start = time.time()
        cam_end = time.time()
        self.Camera_Time(cam_end - self.Camera_Start_Time)

        try:
            boxes, cls_id, conf, masks, r, pad = self.model.infer(self.Frame)

            print("DEBUG boxes:", len(boxes))
            print("DEBUG cls_id:", cls_id)
            print("DEBUG conf:", conf)

            annotated, counts, avg_conf = self.model.draw(self.Frame, boxes, cls_id, conf, masks, r, pad)

            print("DEBUG counts:", counts)
            print("DEBUG avg_conf:", avg_conf)

        except Exception as e:
            annotated = self.Frame
            counts = {}
            avg_conf = 0.0
            logging.exception(e)

        self.last_annotated = annotated
        self.last_counts = counts
        self.last_avg_conf = avg_conf

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.Model_Done.emit(qimg, counts, avg_conf)

        end = time.time()
        self.Total_Time(end - start)

        Config.Time_Wait = False
        Config.Button_Wait = False

    def Update_Result(self, Capture, counts, avg_conf):
        scaled = Capture.scaled(self.Camera_Input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.Camera_Input.setPixmap(QPixmap.fromImage(scaled))

        self.last_avg_conf = avg_conf

        if counts:
            txt = " | ".join([f"{k}: {v}" for k, v in counts.items()])
            self.Class_Label.setText(f"<b><u>Detections</u><br>{txt}")
            self.Score.setText("<b><u>Status</u><br>Defects Found")
            self.Avg_Conf_Label.setText(
                f"<b><u>Average Confidence</u><br>{avg_conf * 100:.2f}%"
            )
        else:
            self.Class_Label.setText("<b><u>Detections</u><br>No defects")
            self.Score.setText("<b><u>Status</u><br>No defects")
            self.Avg_Conf_Label.setText("<b><u>Average Confidence</u><br>N/A")

    def Exposure_Changes(self, value):
        Config.Exposure_Time = int(value)
        self.Exposure_Label.setText(f"<b>Exposure Time: {Config.Exposure_Time}")
        self.Exposure_Edit.clear()

    def Manual_Exposure_Changes(self):
        try:
            value = int(self.Exposure_Edit.text())
            if self.Exposure_Slider.minimum() <= value <= self.Exposure_Slider.maximum():
                Config.Exposure_Time = value
                self.Exposure_Label.setText(f"<b>Exposure Time: {Config.Exposure_Time}")
                self.Exposure_Slider.setValue(value)
            else:
                self.Exposure_Edit.setPlaceholderText("Out of range")
        except Exception:
            self.Exposure_Edit.setPlaceholderText("Integer only")

        self.Exposure_Edit.clear()
        self.Exposure_Edit.clearFocus()

    def Analogue_Changes(self, value):
        Config.Analogue_Gain = int(value)
        self.Analogue_Label.setText(f"<b>Analogue Gain: {Config.Analogue_Gain}")

    def Total_Time(self, total_time):
        t = total_time
        if self.EXP_T is None:
            self.EXP_T = t
        else:
            self.EXP_T = Config.alpha * t + (1 - Config.alpha) * self.EXP_T
        self.System_Processing_Time.setText(
            f"<b><u>Processing Time (seconds)</u><br>{self.EXP_T:.3f}"
        )

    def Camera_Time(self, cam_time):
        t = cam_time
        if self.EXP_T_2 is None:
            self.EXP_T_2 = t
        else:
            self.EXP_T_2 = Config.alpha * t + (1 - Config.alpha) * self.EXP_T_2
        self.Camera_Capturing_Time.setText(
            f"<b><u>Capturing Time (seconds)</u><br>{self.EXP_T_2:.3f}"
        )

    def Resolution_List(self):
        try:
            proc = subprocess.run(
                ["v4l2-ctl", "-d", f"/dev/video{Config.Video_Index}", "--list-formats-ext"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            out = proc.stdout
        except Exception:
            return {}

        resolution = {}
        current_format = None

        for line in out.splitlines():
            fmt = re.search(r"\[\d+\]:\s*'(\w+)'", line)
            if fmt:
                current_format = fmt.group(1)
                resolution[current_format] = []

            if current_format:
                size = re.search(r"Size:\s*Discrete\s*(\d+)x(\d+)", line)
                if size:
                    w3, h3 = size.groups()
                    resolution[current_format].append(f"{w3}x{h3}")

        return resolution

    def Resolution_Changes(self):
        if not hasattr(self, "Camera_List") or not hasattr(self, "Keys"):
            return

        fmt_key = self.Keys[Config.Fmt_Num] if self.Keys else "DEFAULT"
        res_list = self.Camera_List.get(fmt_key, [])
        if not res_list:
            self.Res_Width, self.Res_Height = Config.Default_Size
            self.USB_Cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.Res_Width)
            self.USB_Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.Res_Height)
            return

        Config.Res_Num = max(0, min(Config.Res_Num, len(res_list) - 1))
        self.Res_Width, self.Res_Height = map(int, res_list[Config.Res_Num].split('x'))

        self.USB_Cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.Res_Width)
        self.USB_Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.Res_Height)

    def Format_Changes(self):
        if not hasattr(self, "Keys") or not self.Keys:
            return

        Config.Fmt_Num = max(0, min(Config.Fmt_Num, len(self.Keys) - 1))
        key = self.Keys[Config.Fmt_Num]

        if key == "DEFAULT":
            return

        try:
            self.USB_Cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*key))
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            if hasattr(self, "USB_Cam") and self.USB_Cam and self.USB_Cam.isOpened():
                self.USB_Cam.release()
        except Exception:
            pass
        try:
            GPIO.cleanup()
        except Exception:
            pass
        event.accept()

    def Closing(self):
        self.close()
        sys.exit(0)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.Closing()
        elif event.key() == Qt.Key_T:
            if not Config.Button_Wait and not Config.Time_Wait:
                Config.Button_Wait = True
                threading.Thread(target=self.Trigger_Model, daemon=True).start()
        elif event.key() == Qt.Key_E:
            Config.Res_Num += 1
            self.Resolution_Changes()
        elif event.key() == Qt.Key_Q:
            Config.Res_Num -= 1
            self.Resolution_Changes()
        elif event.key() == Qt.Key_R:
            Config.Fmt_Num += 1
            self.Format_Changes()
        elif event.key() == Qt.Key_W:
            Config.Fmt_Num -= 1
            self.Format_Changes()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    App = QApplication(sys.argv)
    Screen = GUI()
    Screen.showFullScreen()
    sys.exit(App.exec_())