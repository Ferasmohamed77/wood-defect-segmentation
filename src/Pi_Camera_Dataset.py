import sys
import os
import logging
import threading

#Library for GUI Purpose
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QLineEdit
    )
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage
from picamera2 import Picamera2
import cv2

#Library for GPIO Purpose
import RPi.GPIO as GPIO

class Config:		#All setting can be managed here
    #For Model
    Input_Size = (224, 224)		#Size for MobileNet Model. Change if using others
    Button_Wait = False		#Variable used to prevent overloading
    
    #For GUI
    font = QFont("Times New Roman", 15)		#Setting for words
    Logo_Path = r"/home/controleasy/controleasyvisionproject/Images/ControlEasy"		#Path to the LOGO Image
    Bad_Ref_Path = r"/home/controleasy/controleasyvisionproject/Images/Bad"		#Path to the Reference Image (Bad)
    Good_Ref_Path = r"/home/controleasy/controleasyvisionproject/Images/Good"		#Path to the Reference Image (Good)
    
    #For Camera
    Awb_Enable = False		#Enable/Disable Auto White Balance
    Ae_Enable = False		#Enable/Disable Auto Exposure
    Analogue_Gain = 0		#Adjust Camera Brightness	(Only Integer)		
    Exposure_Time = 2550	#Adjust Camera Brightness and Motion Blur	(Only Integer)
    Min_Exposure = 0		#Minimum Exposure Time of this camera
    Max_Exposure = 150000	#Maximum Exposure Time of this camera
    Min_Analogue = 0		#Minimum analogue gain of this camera
    Max_Analogue = 100		#Maximum analogue gain of this camera
    Default_Size = (640, 480)	#Size of camera input/display
    
    Resolution_Switch = False	#A variable to switch resolution    
    
    #For Dataset Development
    Main_Folder = "Wood Defects Dataset(PiCAM)"		#Name of the dataset
    Main_Path = r"/home/controleasy/control_ez_vision_projects"					#The place to store the dataset
    SubFolders = ["train/good","train/bad","val/good","val/bad"]		#The structure of the dataset
    DS_Type = "train"		#Initial type be saved in dataset
    DS_Class = "good"		#Initial class be saved in dataset
    
    #For GPIO
    PIN = 4		#GPIO Pin
    Sensor_Wait = False		#Allow sensor to trigger model for 1 time only
    GPIO.setmode(GPIO.BCM)	#Use GPIO Number System	(GPIO2,3,4......)	#MUST if you're using GPIO Pin Diagram
    GPIO.setup(PIN, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)	#(The GPIO Pin_ , Set as Input, activate internal pull-down)
        
class GUI(QMainWindow):		#Class where all GUI will be involved
    
    Transforms_Done = pyqtSignal(object)		#Setup a variable to detect Trigger_Transforms Function is Completed or not
 
    def __init__(self):		#Function that run once
        super().__init__()		#Initialize the QMainWindow. Must
        Widget = QWidget()		#Variable to be used as QWidget()
        self.setCentralWidget(Widget)   #Set Widget as main content. MUST

        Widget.setFont(Config.font)    #Set all font to be the same
        
        self.Transforms_Done.connect(self.Update_Result)		#Connect to Update_Result Function when Transforms_Done is completed

        self.Camera_Input = self.Make_Label(None, "Panel", 1)			#Use Make_label Function to create QLabel
        self.Camera_Input.setFixedSize(Config.Default_Size[0], Config.Default_Size[1])	#Fix the size of this QLabel
        self.Input_Label = self.Make_Label(f"<b>Camera Input","Sunken", 5)	#Use Make_label Function to create QLabel
        self.Input_Label.setFixedHeight(40)	#Set the Fixed height of this QLabel
        
        Ref_Label = self.Make_Label("<b>Reference", "Sunken", 5)	#Use Make_label Function to create QLabel
        Ref_Label.setFixedHeight(40)	#Set the Fixed height of this QLabel
        
        self.Capture = self.Make_Label(f"<b>N/A", "Panel", 1)						#Use Make_label Function to create QLabel
        self.Capture.setScaledContents(True)										#Make the Image scale with this QLabel
        self.Capture.setFixedSize(Config.Input_Size[0], Config.Input_Size[1])	#Fix size of this QLabel
        
        Capture_Label = self.Make_Label(f"<b>{Config.Input_Size[0]} x {Config.Input_Size[1]}", "Sunken", 5)		#Use Make_label Function to create QLabel
        Capture_Label.setFixedHeight(40)									#Set the fixed height of this QLabel
        
        self.Camera_Resolution = self.Make_Label(None, "Raised", 5)	#Use Make_label Function to create QLabel
        self.Camera_Resolution.setFixedHeight(80)					#Set the Fixed height of this QLabel
        
        self.Type_Label = self.Make_Label(f"<b><u>Type</u><br>{Config.DS_Type}", "Raised", 5)	#Use Make_label Function to create QLabel
        self.Type_Label.setFixedHeight(80)						#Set the Fixed height of this QLabel
        
        self.Class_Label = self.Make_Label(f"<b><u>Class</u><br>{Config.DS_Class}", "Raised", 5)	##Use Make_label Function to create QLabel
        self.Class_Label.setFixedHeight(80)						#Set the Fixed height of this QLabel
        
        self.Path_Label = self.Make_Label(f"<b>Path:</b>", None, 0)
      
        Exit_Button = QPushButton("Press ESC To Quit")			#Create a Push Button
        Exit_Button.setStyleSheet("""						
            QPushButton {
                border: 2px solid white;
                border-radius: 8px;
                padding: 6px;
                background-color: white
                color: white
            }
        """)													#Setup the option
        Exit_Button.clicked.connect(self.Closing)				#Connect to function self.Closing when clicked
        
        EZLogo = QLabel()				#Create a QLabel (Didnt use self.Make_Label because not needed)
        EZLogo.setPixmap(QPixmap(Config.Logo_Path))	#Place Image into this QLabel
        EZLogo.setScaledContents(True)				#Make Image scale with this QLabel
        EZLogo.setMaximumSize(300,100)				#Set the maximum size of this QLabel

        EZLogo2 = QLabel("Machine Vision & Control Automation")		#Create a QLabel (Didnt use self.Make_Label because not needed)
        Text_Size = int(EZLogo2.width() / 10)		#Formula to calculate the font-size	(Approximately)
        EZLogo2.setStyleSheet(f"""
        color: white;
        background-color: black;
        border: 10px outset gray; /*Raised effect */ 
        font-size: {Text_Size}px;
        font-family: Times New Roman;   
        """)									#Setup the option
        EZLogo2.setAlignment(Qt.AlignCenter)	#Align Text at Center

        EZLogo3 = QLabel()					#Create a QLabel (Didnt use self.Make_Label because not needed)
        EZLogo3.setPixmap(QPixmap(Config.Logo_Path))		#Place an Image into this QLabel
        EZLogo3.setScaledContents(True)						#Set the image to scale with this QLabel
        EZLogo3.setMaximumSize(300,100)						#Set the maximum size of this QLabel
    
        self.Exposure_Slider = self.Make_Slider(Config.Min_Exposure, Config.Max_Exposure, Config.Exposure_Time, 1000)	#Use Make_Slider function to create QSlider	(Only Integer Value)
        self.Exposure_Slider.valueChanged.connect(self.Exposure_Changes)		#Trigger the self.Exposure_Changes function if value changes
        
        self.Exposure_Label = self.Make_Label(f"<b>Exposure Time: {Config.Exposure_Time}", "Sunken", 5)		#Use Make_Label function to create QLabel
        self.Exposure_Edit = QLineEdit()		#Create a user input
        self.Exposure_Edit.setPlaceholderText(f"Enter a custom exposure time here ({self.Exposure_Slider.minimum()} to {self.Exposure_Slider.maximum()})")		#Show initial value
        self.Exposure_Edit.setFrame(False)		#Remove its frame
        self.Exposure_Edit.setStyleSheet("""QLineEdit{
        background: transparent;
        font-weight: bold;
        color: black;
        margin-left: 5px;
        margin-right: 5px;
        }""")		#Edit the QLineEdit
        self.Exposure_Edit.setFont(Config.font)		#Set font type and size
        self.Exposure_Edit.returnPressed.connect(self.Manual_Exposure_Changes)	#Detect if user enter the input value.

        self.Analogue_Slider = self.Make_Slider(Config.Min_Analogue, Config.Max_Analogue, Config.Analogue_Gain, 1)	#Use Make_Slider function to create QSlider	(Only Integer Value)
        self.Analogue_Slider.valueChanged.connect(self.Analogue_Changes)		#Trigger the self.Analogue_Changes function if value changes
        self.Analogue_Label = self.Make_Label(f"<b>Analogue Gain: {Config.Analogue_Gain}", "Sunken", 5)	#Use Make_label Function to create QLabel
    
        Row1 = QHBoxLayout()				#Setup a Horizontal Box layut (First Row)
        Row1.addWidget(EZLogo)				#Add QLabel (Widget)
        Row1.addSpacing(50)				#Add fixed empty space
        Row1.addWidget(EZLogo2)				#Add QLabel (Widget)
        Row1.addSpacing(50)				#Add fixed empty space
        Row1.addWidget(EZLogo3)				#Add QLabel (Widget)

        Row2_Left = QVBoxLayout()					#Create a Vertical Box Layout
        Row2_Left.addWidget(self.Input_Label)			#Add QLabel (Widget)
        Row2_Left.addSpacing(10)						#Add fixed empty space
        Row2_Left.addWidget(self.Camera_Input)		#Add QLabel (Widget)
        Row2_Left.addStretch()						#Add as much flexible emty spaces as possible

        Row2_Middle = QVBoxLayout()				#Create a Vertical Box layout
        Row2_Middle.addWidget(Ref_Label)		#Add QLabel (Widget)
        Row2_Middle.addSpacing(10)				#Add fixed empty space
        Row2_Middle.addWidget(self.Capture)			#Add QLabel (Widget)
        Row2_Middle.addWidget(Capture_Label)		#Add QLabel (Widget)
        Row2_Middle.addStretch()				#Add as much flexible empty Spaces as possible

        Row2_Right = QVBoxLayout()						#Create a Vertical Box Layout
        Row2_Right.addWidget(self.Camera_Resolution)	#Add QLabel (Widget)
        Row2_Right.addSpacing(30)						#Add fixed spacing
        Row2_Right.addWidget(self.Type_Label)			#Add QLabel (Widget)
        Row2_Right.addSpacing(30)						#Add fixed spacing
        Row2_Right.addWidget(self.Class_Label)			#Add QLabel (Widget)
        Row2_Right.addSpacing(30)					#Add fixed empty space
        Row2_Right.addWidget(self.Exposure_Slider)	#Add QSlider
        Row2_Right.addWidget(self.Exposure_Label)	#Add QLabel
        Row2_Right.addWidget(self.Exposure_Edit)		#Add User Input (QLineEdit)
        Row2_Right.addSpacing(30)					#Add fixed empty space
        Row2_Right.addWidget(self.Analogue_Slider)	#Add QSlider
        Row2_Right.addWidget(self.Analogue_Label)	#Add QLabel
        Row2_Right.addStretch()							#Add as much flexible empty Spaces as possible
        
        Row2 = QHBoxLayout()							#Create a horizontal box layout (Second Row)
        Row2.addLayout(Row2_Left)						#Add Row2_Left Layout (Nested Layout)
        Row2.addSpacing(100)							#Add fixed spacing
        Row2.addLayout(Row2_Middle)						#Add Row2_Middle Layout (Nested Layout)
        Row2.addSpacing(100)							#Add fixed spacing
        Row2.addLayout(Row2_Right)						#Add Row2_Right Layout (Nested Layout)
        
        Row3 = QHBoxLayout()				#Create a Horizontal Box Layout
        Row3.addWidget(self.Path_Label)		#Add QLabel
        Row3.addStretch()					#Add as much flexible empty Spaces as possible
        Row3.addWidget(Exit_Button)			#Add QPushButton (Widget)        
        
        GUIArrange = QVBoxLayout()								#Create the MAIN Layout (Vertical)
        GUIArrange.setContentsMargins(20, 10, 20, 20)			#Add spacing between layout border and content (left, top, right, bottom)
        GUIArrange.addLayout(Row1)								#Add the Row1 Layout (Nested Layout)
        GUIArrange.addSpacing(20)								#Add fixed spacing
        GUIArrange.addLayout(Row2)								#Add the Row2 Layout (Nested Layout)
        GUIArrange.addStretch()									#Add as much flexible empty Spaces as possible
        GUIArrange.addLayout(Row3)								#Add the Row3 Layout (Nested Layout)
        
        Widget.setLayout(GUIArrange)			#Attach the Main Layout to Widget (MUST)
        
        self.Camera()							#Run the self.Camera() Function
        self.Create_Folder()		#Create the dataset folder        
    
        self.Timer = QTimer()		#Setup Timer
        self.Timer.start(30)		#30ms / 30 fps (loop)
        self.Timer.timeout.connect(self.Update_Frame)		#Connect to Function self.Update_Frame whenever times up    
    
    def Make_Label(self, Text, Style, Line):		#Function to create QLabel easier
        label = QLabel(Text)		#Your Text
        
        if Style == "Sunken":
            label.setFrameStyle(QLabel.Panel | QLabel.Sunken)	#Customize Frame Style
        elif Style == "Raised":
            label.setFrameStyle(QLabel.Panel | QLabel.Raised)
        elif Style == "Panel":
            label.setFrameStyle(QLabel.Panel)
            
        label.setLineWidth(Line)		#Adjust Line Width of Frame
        label.setAlignment(Qt.AlignCenter)	#Adjust Alignment of the Text
        return label		#Apply the QLabel
    
    def Make_Slider(self, min, max, initial, gap):			#Function to create HORIZONTAL Slider	(Only Integer value)
        slider = QSlider(Qt.Horizontal, self)		#Create a slider
        slider.setRange(min, max)					#Set the range	(Only Integer)
        slider.setValue(initial)					#Set the initial value	(Only Integer)
        slider.setTickPosition(QSlider.TicksBelow)	#Set the mark position
        slider.setTickInterval(gap)					#Set the range between each mark	(Only Integer)
        return slider		#Apply the slider    
    
    def Camera(self):		#Function for Camera Setup
        self.PiCamera = None	#Initial Condition
            
        try:
            self.PiCamera = Picamera2()		#Connect to Raspberry Pi camera
            self.Best_Resolution = self.PiCamera.sensor_modes[0]["size"]	#Get the resolution of the pi camera	#Access to first mode [0] and search for 'size'	#this is a list with dictionary
            
            self.PiCamera.configure(self.PiCamera.create_preview_configuration(main={"format": "BGR888"}))	#Setup Input to be in BGR Format, 888 = 8 bits or 255 for each channel
            #Create live view and output the image in 8-bit RGB		#Default Resolution
            self.PiCamera.set_controls({		#Setup Camera Setting
                "AwbEnable": Config.Awb_Enable,		#Enable/Disable Auto White Balance
                "AeEnable": Config.Ae_Enable,		#Enable/Disable Auto Exposure
                "AnalogueGain": Config.Analogue_Gain,		#Adjust Brightness
                "ExposureTime": Config.Exposure_Time		#Adjust Brightness, motion blur
                })
            #print(self.PiCamera.camera_controls)		#Uncomment it to see the parameter of the camera	#Comment it if no use, may cause some issue
            #print(self.PiCamera.sensor_modes)			#Uncomment it to see the parameter of the camera	#Comment it if no use, may cause some issue
            #print(self.PiCamera.camera_properties)		#Uncomment it to see the parameter of the camera	#Comment it if no use, may cause some issue
            self.PiCamera.start()		#Start the camera
        
            Extract_Time = self.PiCamera.camera_controls["ExposureTime"]	#Extract the exposure time range from camera
            Config.Min_Exposure = Extract_Time[0]	#Minimum exposure time
            Config.Max_Exposure = Extract_Time[2]	#Not really max value, this is default value)
            self.Exposure_Slider.setRange(Config.Min_Exposure, Config.Max_Exposure)	#Update it to latest version
            self.Exposure_Edit.setPlaceholderText(f"Enter a custom exposure time here ({self.Exposure_Slider.minimum()} to {self.Exposure_Slider.maximum()})")		#Update too
        
        except Exception as e:	#Trigger if no camera connected
            self.Input_Label.setText("Error")	#Change Text of QLabel
            self.Camera_Input.setText("Error")		#Change Text of QLabel
            logging.warning(f"PiCamera is not connected or accessible.")	#Show error        
            return				#Close the function
    
    def Update_Frame(self):		#Function to camera livestream (Continuous)
        self.Frame = self.PiCamera.capture_array()	#Take 1 frame as Numpy array , Image format is reverse of configuration, RGB format
        #print(self.Frame[0,0])		#Uncomment it to see the channel value to know RGB / BGR

        h, w,ch = self.Frame.shape		#(Image height, Image Width, Image Channels)
        bytes_per_line = ch * w		#Calculate the number of bytes per row fpr all channels
        self.QImg = QImage(self.Frame.data, w, h, bytes_per_line, QImage.Format_RGB888)	#(Pixel data,_,_,_,#RGB888 mean view first channel as R, second as G, third as B, 8 bits or 255 for each channel])
        Scaled_Image = self.QImg.scaled(self.Camera_Input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)		#Scale the image to fit the QLabel Size
        self.Camera_Input.setPixmap(QPixmap.fromImage(Scaled_Image))		#Place Image into QLabel (Continuously)

        self.Camera_Resolution.setText(f"<b><u>Resolution</u><br>{w} x {h}")	#Change Text of QLabel
        
        if GPIO.input(Config.PIN) == GPIO.HIGH and not Config.Sensor_Wait:		#Trigger function once if sensor is HIGH
            Config.Sensor_Wait = True			#Make sure function only run once
            threading.Thread(target=self.Trigger_Transforms).start()
        elif GPIO.input(Config.PIN) == GPIO.LOW:			#Reset if sensor is LOW
            Config.Sensor_Wait = False			#Reset
            
    def Trigger_Transforms(self):		#Function to run the transforms / preprocessing, 	#channel is a must if using GPIO, it represent GPIO Pin Number which will be pass down by the GPIO.detect
        h2, w2, ch2 = self.Frame.shape	#(Image height, Image Width, Image Channels)  
        Scale_Factor = Config.Input_Size[0] / min(h2, w2)		#Get the smaller side	Calculate scale factor
        new_h, new_w = int(h2 * Scale_Factor), int(w2 * Scale_Factor)	#Get the new size of image based on scale factor (Keep Ratio)
        
        Resize = cv2.resize(self.Frame, (new_w, new_h), interpolation = cv2.INTER_LANCZOS4)		#Resize based on new size and use LANCZOS4 filter for better image

        Vertical_Cut = (new_h - Config.Input_Size[0]) // 2		#Calculate pixels to be remove to fit into desired size, //2 because it split up and down
        Horizontal_Cut = (new_w - Config.Input_Size[0]) //2		#Calculate pixels to be remove to fit into desired size, //2 because it split left and right
        Crop = Resize[Vertical_Cut:(Vertical_Cut + Config.Input_Size[0]), Horizontal_Cut:(Horizontal_Cut + Config.Input_Size[0])]		#Crop the image to the desired size
        #Verticle:(Vertical+Size) mean the pixel we want to keep, like 37:261, pixel 37 to 261 is the pixels to keep, other remove
        
        self.Result = Crop.copy()		#Copy Image 
        h3, w3, ch3 = self.Result.shape		#(Image height, Image width, Image Channels)
        bytes_per_line_3 = ch3 * w3		#Calculate the number of bytes per row fpr all channels
        self.QImg_Capture = QImage(self.Result.data, w3, h3, bytes_per_line_3, QImage.Format_RGB888)	#(Pixel data,_,_,_,#RGB888 mean view first channel as R, second as G, third as B, 8 bits or 255 for each channel])
        
        self.Transforms_Done.emit(self.QImg_Capture)		#Send signal back to GUI along with these data
        
        self.Save_Image()		#Call the Save_Image Function

    def Update_Result(self, Capture):		#Function to update the latest data to display
        self.Capture.setPixmap(QPixmap.fromImage(Capture))				#Place Image into QLabel
            
    def Create_Folder(self):
        self.Dataset_Folder = os.path.join(Config.Main_Path, Config.Main_Folder)		#Setup the path
        
        for sub in Config.SubFolders:								#Create the folder one by one
           Folders = os.path.join(self.Dataset_Folder, sub)		#Setup the path
           os.makedirs(Folders, exist_ok = True)			#Create dataset and its subfolders according to path, do nothing if already exists
        
    def Save_Image(self):				#Function to save image
        if hasattr(self, "Frame"):		#Detect the existance of self.Frame (aka Camera)
            Folder_Path = os.path.join(self.Dataset_Folder, Config.DS_Type, Config.DS_Class)	#Setup the path for saving image
            
            Exist_Image = []	#Create an empty list
            for f in os.listdir(Folder_Path):			#Loop through every filename in the folder
                if f.lower().endswith((".jpg", ".jpeg", ".png"))	:#Check if the file is an image
                    Exist_Image.append(f)		#Record it into the list
            
            Count = len(Exist_Image) +1		#Calculate how many image existed
            filename = f"{Config.DS_Class} {Count}" + ".jpg"		#Name of the saved image. #jpg format
            Save_Path = os.path.join(Folder_Path, filename)	#Path for the saved image
            cv2.imwrite(Save_Path, cv2.cvtColor(self.Result, cv2.COLOR_RGB2BGR))		#Convert image to BGR Format and save it.
            #BGR2RGB = RGB2BGR Its different name but same function god damn it, it just swap the Red and Blue channel
            
            self.Path_Label.setText(f"<b>Path:</b> {Save_Path}")		#Show the saved path in this QLabel
            Config.Button_Wait = False		#Reset
    
    def Exposure_Changes(self, value):		#Function that update exposure time based on slider
        Config.Exposure_Time = value		#Change to latest exposure time
        
        self.PiCamera.set_controls({		#Update Camera Setting
            "AwbEnable": Config.Awb_Enable,		#Enable/Disable Auto White Balance
            "AeEnable": Config.Ae_Enable,		#Enable/Disable Auto Exposure
            "AnalogueGain": Config.Analogue_Gain,		#Adjust Brightness
            "ExposureTime": Config.Exposure_Time		#Adjust Brightness, motion blur
            })
        
        self.Exposure_Label.setText(f"<b>Exposure Time: {Config.Exposure_Time}")	#Update the current exposure time
        self.Exposure_Edit.clear()	#Reset
        
    def Manual_Exposure_Changes(self):
        try:		#Wont shut down program if failed to run
            value = int(self.Exposure_Edit.text())	#Failed if input not integer	#Record it
            
            if self.Exposure_Slider.minimum() <= value <= self.Exposure_Slider.maximum():	#Check if input is within the range
                Config.Exposure_Time = value	#Change exposure time
                self.Exposure_Label.setText(f"<b>Exposure Time: {Config.Exposure_Time}")	#Update the current exposure time
                self.Exposure_Slider.setValue(value)	#Update the exposure time slider
            else:
                self.Exposure_Error("Out of range")		#Run the function to show error
                
        except:		#Trigger if try: failed
            self.Exposure_Error("Integer Value Only")	#Run the function to show error
        
        self.Exposure_Changes(Config.Exposure_Time)
        self.Exposure_Edit.clear()	#Reset Input Box
        self.Exposure_Edit.clearFocus()	#Give keyboard control back to main window
    
    def Exposure_Error(self, message):			#Function to show error
        self.Exposure_Edit.setReadOnly(True)	#Set Input box as read only / disable user input
        self.Exposure_Edit.setPlaceholderText(message)	#Show error
        QTimer.singleShot(2000, lambda: (self.Exposure_Edit.setPlaceholderText(f"Enter a custom exposure time here ({self.Exposure_Slider.minimum()} to {self.Exposure_Slider.maximum()})"),
                                         self.Exposure_Edit.setReadOnly(False)))	#Reset Placeholder Text after 2 second

    def Analogue_Changes(self, value):		#Function to update the analogue gain value
        Config.Analogue_Gain = value	#Update the analogue gain
        self.Analogue_Label.setText(f"<b>Analogue Gain: {Config.Analogue_Gain}")	#Update QLabel to show latest analogue gain
    
        self.PiCamera.set_controls({		#Update Camera Setting
            "AwbEnable": Config.Awb_Enable,		#Enable/Disable Auto White Balance
            "AeEnable": Config.Ae_Enable,		#Enable/Disable Auto Exposure
            "AnalogueGain": Config.Analogue_Gain,		#Adjust Brightness
            "ExposureTime": Config.Exposure_Time		#Adjust Brightness, motion blur
            })    
    
    def Resolution(self):
        if not Config.Resolution_Switch:		#Switch between 2 resolution
            self.PiCamera.stop()		#Stop the Pi Camera
            self.PiCamera.configure(self.PiCamera.create_preview_configuration(main={"format": "BGR888", "size": self.Best_Resolution}))		#Setup new resolution
            self.PiCamera.start()		#Start the Pi Camera
            Config.Resolution_Switch = True
        else:
            self.PiCamera.stop()		#Stop the Pi Camera
            self.PiCamera.configure(self.PiCamera.create_preview_configuration(main={"format": "BGR888"}))		#Reset to default resolution
            self.PiCamera.start()		#Start the Pi Camera
            Config.Resolution_Switch = False
            
        Update = self.PiCamera.capture_array()	#Take 1 frame as Numpy array	#Somehow image here is BGR but previous RGB formats is a MUST
        h3, w3, ch3 = Update.shape		#Get the dimension (height, width, channels)      #Update the latest width x height 
        self.Camera_Resolution.setText(f"<b><u>Resolution</u><br>{w3} x {h3}")	#Update the resolution in QLabel    
    
    def closeEvent(self, event):		#Fixed Name, Function of Qt triggerd when window is shutting down
        if self.PiCamera:			#check camera is on or not
            self.PiCamera.stop()	#Stop the camera
            
        event.accept()				#Allow Window to close
        
    def Closing(self):		#Function for ESC Button
        self.close()		#Close Window
        sys.exit(0)			#Close Entire Program and return code 0
    
    def keyPressEvent(self, event):		#Fixed name, Function to detect keyboard
        if event.key() == Qt.Key_Escape:		#Detect ESC Button
            self.Closing()			#Trigger Closing() Function
            
        if event.key() == Qt.Key_T:			#Detect T Button
            if not Config.Button_Wait:		#Run the content when detecting False
                Config.Button_Wait = True		#Make sure it finish first before running again
                threading.Thread(target=self.Trigger_Transforms).start()
                
        elif event.key() == Qt.Key_G:			#Detect G Button
            self.Resolution()
            
        if event.key() == Qt.Key_Q:				#Detect Q Button
            if Config.DS_Type == "train":		#Change dataset type from train to val
                Config.DS_Type = "val"
            else:
                Config.DS_Type = "train"		#Change dataset type from val to train
            self.Type_Label.setText(f"<b><u>Type</u><br>{Config.DS_Type}")	#Change Text of QLabel
            
        if event.key() == Qt.Key_E:				#Detect E Button
            if Config.DS_Class == "good":		#Change dataset class from good to bad
                Config.DS_Class = "bad"
            else:
                Config.DS_Class = "good"		#Change dataset class form bad to good
            self.Class_Label.setText(f"<b><u>Class</u><br>{Config.DS_Class}")	#Change Text of QLabel
            
if __name__ == "__main__":		#Run the code if file executed successfully
    App = QApplication(sys.argv) #Main Qt Application. Must if using PyQt App. Manage all GUI
    Screen = GUI()  #Create GUI
    Screen.showFullScreen()   #Show in Full Screen
    
    sys.exit(App.exec_()) #System exit when the loop finished / close by user (App.exec_()
