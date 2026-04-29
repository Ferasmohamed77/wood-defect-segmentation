import sys
import os
import time
import threading
import numpy as np
import cv2
import onnxruntime as ort
import RPi.GPIO as GPIO

from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, 
    QVBoxLayout, QHBoxLayout, QFrame, QPushButton, QSlider, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage, QKeyEvent

from picamera2 import Picamera2

# ---------------- CONFIG ----------------
class Config:
    Model_Path = r"/home/controleasy/controleasyvisionproject/best.onnx"
    Model_ImgSz = 640
    Conf_Thres = 0.25 
    IoU_Thres = 0.60  
    Classes = ["Big Hole", "Scratches", "Small Hole"]

    PIN = 4 
    font_main = QFont("Arial", 11)
    font_title = QFont("Arial", 14, QFont.Bold)
    
    Default_Exposure = 20000 
    Default_Gain = 1.0
    
    Time_Wait = False
    Sensor_Wait = False

# ---------------- MODEL ----------------
class SegModel:
    def __init__(self):
        self.session = ort.InferenceSession(Config.Model_Path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        r = min(Config.Model_ImgSz / h, Config.Model_ImgSz / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = (Config.Model_ImgSz - new_unpad[0]) / 2, (Config.Model_ImgSz - new_unpad[1]) / 2
        
        resized = cv2.resize(frame_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(resized, int(round(dh - 0.1)), int(round(dh + 0.1)), 
                                 int(round(dw - 0.1)), int(round(dw + 0.1)), 
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_in = np.transpose(img_in, (2, 0, 1))[None]
        
        outs = self.session.run(None, {self.input_name: img_in})
        pred = np.squeeze(outs[0])
        if pred.ndim == 1: return [], [], [], r, (dw, dh)
        if pred.shape[0] < pred.shape[1]: pred = pred.T

        conf = np.max(pred[:, 4:4+len(Config.Classes)], axis=1)
        mask_idx = conf > Config.Conf_Thres
        if not np.any(mask_idx): return [], [], [], r, (dw, dh)
        
        pred = pred[mask_idx]
        conf = conf[mask_idx]
        boxes = np.empty_like(pred[:, :4])
        boxes[:, 0], boxes[:, 1] = pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2
        boxes[:, 2], boxes[:, 3] = pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2
        cls_ids = np.argmax(pred[:, 4:4+len(Config.Classes)], axis=1)
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), Config.Conf_Thres, Config.IoU_Thres)
        if len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices], cls_ids[indices], conf[indices], r, (dw, dh)
        return [], [], [], r, (dw, dh)

# ---------------- GUI ----------------
class GUI(QMainWindow):
    Model_Done = pyqtSignal(object, dict, float, float, float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Vision & Control Automation")
        self.setStyleSheet("background-color: #E0E0E0;")
        self.model = SegModel()
        self.last_annotated = None

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(Config.PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # 1. TOP TITLE BAR
        self.Top_Title = QLabel("Machine Vision & Control Automation")
        self.Top_Title.setAlignment(Qt.AlignCenter)
        self.Top_Title.setStyleSheet("background-color: black; color: white; padding: 20px;")
        self.Top_Title.setFont(Config.font_title)
        self.layout.addWidget(self.Top_Title)

        # 2. MIDDLE CONTENT AREA
        self.mid_container = QWidget()
        self.mid_layout = QHBoxLayout(self.mid_container)
        
        # Left Side: Camera
        self.cam_vbox = QVBoxLayout()
        self.cam_vbox.addWidget(self.make_label("Camera Input", sunken=True))
        self.Camera_Input = QLabel()
        self.Camera_Input.setFixedSize(640, 480)
        self.Camera_Input.setStyleSheet("background-color: #000; border: 1px solid #999;")
        self.cam_vbox.addWidget(self.Camera_Input)
        self.mid_layout.addLayout(self.cam_vbox)

        # Right Side: Info Labels
        self.info_vbox = QVBoxLayout()
        self.Res_Label = self.make_label("<b><u>Resolution</u></b><br>YUYV: 640 x 480 [0]")
        self.CapTime_Label = self.make_label("<b><u>Capturing Time (seconds)</u></b><br>N/A")
        self.ProcTime_Label = self.make_label("<b><u>Processing Time (seconds)</u></b><br>N/A")
        self.Detect_Label = self.make_label("<b><u>Detections</u></b><br>N/A")
        self.Status_Label = self.make_label("<b><u>Status</u></b><br>Ready")
        self.Conf_Label = self.make_label("<b><u>Average Confidence</u></b><br>N/A")
        self.ShowConf_Btn = QPushButton("Show Average Confidence")
        self.ShowConf_Btn.setStyleSheet("background-color: #DDD; height: 35px; border: 1px solid #999;")
        
        for widget in [self.Res_Label, self.CapTime_Label, self.ProcTime_Label, 
                       self.Detect_Label, self.Status_Label, self.Conf_Label, self.ShowConf_Btn]:
            self.info_vbox.addWidget(widget)
        self.mid_layout.addLayout(self.info_vbox)
        self.layout.addWidget(self.mid_container)

        # 3. SLIDERS SECTION (Fixed Height to ensure visibility)
        self.slider_frame = QFrame()
        self.slider_frame.setFixedHeight(220)
        self.slider_vbox = QVBoxLayout(self.slider_frame)
        
        self.Exp_Slider = QSlider(Qt.Horizontal)
        self.Exp_Slider.setRange(1, 100000)
        self.Exp_Slider.setValue(Config.Default_Exposure)
        self.Exp_Title = self.make_label(f"Exposure Time: {Config.Default_Exposure}")
        
        self.Exp_Input = QLineEdit()
        self.Exp_Input.setPlaceholderText("Enter exposure (1 to 100000)")
        self.Exp_Input.setStyleSheet("background-color: white; border: 1px solid gray; padding: 2px;")
        
        self.Gain_Slider = QSlider(Qt.Horizontal)
        self.Gain_Slider.setRange(1, 16)
        self.Gain_Slider.setValue(int(Config.Default_Gain))
        self.Gain_Title = self.make_label(f"Analogue Gain: {Config.Default_Gain}")

        for s_widget in [self.Exp_Slider, self.Exp_Title, self.Exp_Input, self.Gain_Slider, self.Gain_Title]:
            self.slider_vbox.addWidget(s_widget)
        self.layout.addWidget(self.slider_frame)

        # 4. BOTTOM EXIT BUTTON (Text Button)
        self.Bottom_Frame = QFrame()
        self.Bottom_Frame.setStyleSheet("background-color: #CCC;")
        self.bottom_layout = QHBoxLayout(self.Bottom_Frame)
        
        self.Exit_Btn = QPushButton("Press ESC To Quit")
        self.Exit_Btn.setFont(Config.font_main)
        self.Exit_Btn.setStyleSheet("QPushButton { border: none; background: transparent; color: black; padding: 10px; }")
        self.Exit_Btn.clicked.connect(self.close)
        
        self.bottom_layout.addWidget(self.Exit_Btn, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.Bottom_Frame)

        # Connections
        self.Model_Done.connect(self.Update_Result)
        self.Exp_Slider.valueChanged.connect(self.update_cam_params)
        self.Gain_Slider.valueChanged.connect(self.update_cam_params)
        self.Exp_Input.returnPressed.connect(self.manual_exp_entry)

        self.picam = Picamera2()
        self.picam.configure(self.picam.create_video_configuration(main={"size": (640, 480)}))
        self.picam.start()
        self.update_cam_params()

        self.timer = QTimer()
        self.timer.timeout.connect(self.Camera_Loop)
        self.timer.start(30)

    def make_label(self, text, sunken=False):
        lbl = QLabel(text); lbl.setAlignment(Qt.AlignCenter); lbl.setFont(Config.font_main)
        lbl.setFrameStyle(QFrame.Panel | (QFrame.Sunken if sunken else QFrame.Raised))
        lbl.setStyleSheet("background-color: #F8F8F8; border: 1px solid #AAA; padding: 2px;")
        return lbl

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape: self.close()
        elif event.key() == Qt.Key_T: threading.Thread(target=self.Run_Inference, daemon=True).start()
        elif event.key() == Qt.Key_C: self.last_annotated = None; self.Status_Label.setText("<b><u>Status</u></b><br>Ready")

    def update_cam_params(self):
        exp, gain = self.Exp_Slider.value(), float(self.Gain_Slider.value())
        self.Exp_Title.setText(f"Exposure Time: {exp}"); self.Gain_Title.setText(f"Analogue Gain: {gain}")
        try: self.picam.set_controls({"ExposureTime": exp, "AnalogueGain": gain})
        except: pass

    def manual_exp_entry(self):
        try: self.Exp_Slider.setValue(int(self.Exp_Input.text()))
        except: pass

    def Camera_Loop(self):
        frame_rgb = self.picam.capture_array()
        self.current_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        display_frame = self.last_annotated if self.last_annotated is not None else self.current_frame
        self.Display_Image(display_frame)

        if GPIO.input(Config.PIN) == GPIO.HIGH and not Config.Sensor_Wait:
            Config.Sensor_Wait = True
            threading.Thread(target=self.Run_Inference, daemon=True).start()
        elif GPIO.input(Config.PIN) == GPIO.LOW:
            Config.Sensor_Wait = False

    def Run_Inference(self):
        if Config.Time_Wait: return
        Config.Time_Wait = True
        t_cap_start = time.time()
        frame_to_process = self.current_frame.copy()
        cap_time = time.time() - t_cap_start 
        
        t_proc_start = time.time()
        boxes, cls_ids, confs, r, pad = self.model.infer(frame_to_process)
        annotated = frame_to_process.copy()
        counts = {}
        avg_conf = 0.0
        
        if len(boxes) > 0:
            avg_conf = float(np.mean(confs))
            for i in range(len(boxes)):
                label = Config.Classes[cls_ids[i]]
                counts[label] = counts.get(label, 0) + 1
                b = (boxes[i] - [pad[0], pad[1], pad[0], pad[1]]) / r
                cv2.rectangle(annotated, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.putText(annotated, label, (int(b[0]), int(b[1]-10)), 0, 0.6, (0, 255, 0), 2)
        
        proc_time = time.time() - t_proc_start
        self.last_annotated = annotated
        self.Model_Done.emit(annotated, counts, avg_conf, cap_time, proc_time)
        Config.Time_Wait = False

    def Update_Result(self, frame, counts, conf, cap_t, proc_t):
        self.Display_Image(frame)
        res_txt = ", ".join([f"{k}:{v}" for k,v in counts.items()]) if counts else "None"
        self.Detect_Label.setText(f"<b><u>Detections</u></b><br>{res_txt}")
        self.Status_Label.setText(f"<b><u>Status</u></b><br>{'Defect Found' if counts else 'Ready'}")
        self.Conf_Label.setText(f"<b><u>Average Confidence</u></b><br>{conf*100:.1f}%")
        self.CapTime_Label.setText(f"<b><u>Capturing Time (seconds)</u></b><br>{cap_t:.4f}")
        self.ProcTime_Label.setText(f"<b><u>Processing Time (seconds)</u></b><br>{proc_t:.4f}")

    def Display_Image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.Camera_Input.setPixmap(QPixmap.fromImage(qimg).scaled(640, 480, Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.picam.stop(); GPIO.cleanup(); event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv); window = GUI(); window.showFullScreen(); sys.exit(app.exec_())