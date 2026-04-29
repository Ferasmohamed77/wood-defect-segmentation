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

#Library for Camera Purpose
import cv2			#Must be OpenCV-Python_Headless because OpenCV-Python will load other GUI which then cause error when using PyQt5
import subprocess	#Run external command from Python
import re			#(Regular Expression)

#Library for GPIO Purpose
import RPi.GPIO as GPIO

class Config:		#All setting can be managed here
    #For Model
    Input_Size = (224, 224)		#Size for MobileNet Model. Change if using others
    Button_Wait = False		#Variable used to prevent overloading
    
    #For GUI
    font = QFont("Times New Roman", 15)		#Setting for words
    Logo_Path = r"/home/controleasy/controleasyvisionproject/Images/ControlEasy"		#Path to the LOGO Image
    Bad_Ref_Path = r"/home/controleasy/control_ez_vision_projects/Images/Bad.jpg"		#Path to the Reference Image (Bad)
    Good_Ref_Path = r"/home/controleasy/control_ez_vision_projects/Images/Good.jpg"		#Path to the Reference Image (Good)
    
    #For Camera
    Awb_Enable = False		#Enable/Disable Auto White Balance
    Ae_Enable = False		#Enable/Disable Auto Exposure
    Analogue_Gain = 1		#Adjust Camera Brightness	(Only Integer)		
    Exposure_Time = 100		#Adjust Camera Brightness and Motion Blur	(Only Integer) (initial value)
    Min_Exposure = 1		#Minimum Exposure Time of this camera
    Max_Exposure = 1000		#Maximum Exposure Time of this camera
    Min_Analogue = 0		#Minimum analogue gain of this camera
    Max_Analogue = 125		#Maximum analogue gain of this camera		#I set half only, Not really need to be maximum
    Video_Index = 0 #The location of camera		#May be different if you plug in multiple camera
    Default_Size = (640, 480)	#Size of camera input/display
    
    Res_Num = 0		#Default Resolution
    Fmt_Num = 0		#Default Format

    #For Dataset Development
    Main_Folder = "Wood Defects Dataset(USBwebcam)"		#Name of the dataset
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
        Input_Label = self.Make_Label(f"<b>Camera Input","Sunken", 5)	#Use Make_label Function to create QLabel
        Input_Label.setFixedHeight(40)	#Set the Fixed height of this QLabel
        
        Ref_Label = self.Make_Label("<b>Reference", "Sunken", 5)	#Use Make_label Function to create QLabel
        Ref_Label.setMinimumWidth(Config.Input_Size[0])	#Set the Minimum width of this QLabel
        Ref_Label.setFixedHeight(40)	#Set the Fixed height of this QLabel
        
        self.Capture = self.Make_Label(f"<b>N/A", "Panel", 1)						#Use Make_label Function to create QLabel
        self.Capture.setFixedSize(Config.Input_Size[0], Config.Input_Size[1])	#Fix the size of this QLabel
        
        Capture_Label = self.Make_Label(f"<b>{Config.Input_Size[0]} x {Config.Input_Size[1]}", "Sunken", 5)		#Use Make_label Function to create QLabel
        Capture_Label.setFixedSize(Config.Input_Size[0], 40)				#Set a fixed size of this QLabel
        
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
        Text_Size = int(EZLogo2.width() / 15)		#Formula to calculate the font-size	(Approximately)
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
        Row1.addSpacing(50)					#Add fixed spacing
        Row1.addWidget(EZLogo2)				#Add QLabel (Widget)
        Row1.addSpacing(50)					#Add fixed spacing
        Row1.addWidget(EZLogo3)				#Add QLabel (Widget)

        Row2_Left = QVBoxLayout()					#Create a Vertical Box Layout
        Row2_Left.addWidget(Input_Label)			#Add QLabel (Widget)
        Row2_Left.addSpacing(10)						#Add fixed spacing
        Row2_Left.addWidget(self.Camera_Input)		#Add QLabel (Widget)
        Row2_Left.addStretch()						#Add as much flexible emty spaces as possible

        Row2_Middle = QVBoxLayout()				#Create a Vertical Box layout
        Row2_Middle.addWidget(Ref_Label)		#Add QLabel (Widget)
        Row2_Middle.addSpacing(10)					#Add fixed Spacing
        Row2_Middle.addWidget(self.Capture)			#Add QLabel (Widget)
        Row2_Middle.addWidget(Capture_Label)		#Add QLabel (Widget)
        Row2_Middle.addStretch()					#Add as much flexible emty spaces as possible

        Row2_Right = QVBoxLayout()						#Create a Vertical Box Layout
        Row2_Right.addWidget(self.Camera_Resolution)	#Add QLabel (Widget)
        Row2_Right.addSpacing(30)						#Add fixed spacing
        Row2_Right.addWidget(self.Type_Label)			#Add QLabel (Widget)
        Row2_Right.addSpacing(30)						#Add fixed spacing
        Row2_Right.addWidget(self.Class_Label)			#Add QLabel (Widget)
        Row2_Right.addSpacing(30)						#Add fixed spacing
        Row2_Right.addWidget(self.Exposure_Slider)	#Add QSlider
        Row2_Right.addWidget(self.Exposure_Label)	#Add QLabel
        Row2_Right.addWidget(self.Exposure_Edit)		#Add User Input (QLineEdit)
        Row2_Right.addSpacing(30)					#Add fixed empty space
        Row2_Right.addWidget(self.Analogue_Slider)	#Add QSlider
        Row2_Right.addWidget(self.Analogue_Label)	#Add QLabel           
        Row2_Right.addStretch()							#Add as much flexible empty Spaces as possible
        
        Row2 = QHBoxLayout()							#Create a horizontal box layout (Second Row)
        Row2.addLayout(Row2_Left)						#Add Row2_Left Layout (Nested Layout)
        Row2.addSpacing(50)							#Add fixed spacing
        Row2.addLayout(Row2_Middle)						#Add Row2_Middle Layout (Nested Layout)
        Row2.addSpacing(50)							#Add fixed spacing
        Row2.addLayout(Row2_Right)						#Add Row2_Right Layout (Nested Layout)
        
        Row3 = QHBoxLayout()				#Create a Vertical Box Layout
        Row3.addWidget(self.Path_Label)
        Row3.addStretch()
        Row3.addWidget(Exit_Button)		#Add QPushButton (Widget)
        
        GUIArrange = QVBoxLayout()								#Create the MAIN Layout (Vertical)
        GUIArrange.setContentsMargins(20, 10, 20, 20)			#Add spacing between layout border and content (left, top, right, bottom)
        GUIArrange.addLayout(Row1)								#Add the Row1 Layout (Nested Layout)
        GUIArrange.addSpacing(20)								#Add fixed spacing
        GUIArrange.addLayout(Row2)								#Add the Row2 Layout (Nested Layout)
        GUIArrange.addSpacing(20)								#Add fixed spacing
        GUIArrange.addLayout(Row3)								#Add the Row3 Layout (Nested Layout)
        
        Widget.setLayout(GUIArrange)			#Attach the Main Layout to Widget (MUST)
        
        self.Create_Folder()		#Create the dataset folder  
    
        self.Camera()					#Run the self.Camera() Function
        self.Timer = QTimer()			#Setup a timer
        self.Timer.start(30)			#30ms / 30FPS	(Loop)
        self.Timer.timeout.connect(self.Update_Camera)		#Run this function when timer finished
    
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
        self.USB_Cam = cv2.VideoCapture(Config.Video_Index, cv2.CAP_V4L2)	#Open the camera(Index) using V4L2 driver
        
        if not os.path.exists(f"/dev/video{Config.Video_Index}"):		#Check if the camera is not physically connected / wrong Index
            logging.warning(f"USB Webcam is not connected in /dev/video{Config.Video_Index}")		#Print out error
            return		#Close function
        elif not self.USB_Cam.isOpened():		#Check if the camera is connected but not accessible
            logging.warning(f"USB Webcam cannot be accessed")		#Print out error
            return		#Close function
        
        self.Camera_List = self.Resolution_List()	#Run the function Resolution_List and get the Dictionary {}
        self.Keys = list(self.Camera_List.keys())	#Record the keys of the dictionary
                
        self.USB_Cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)		#Set to Manual Exposure Time Mode
        self.Resolution_Changes()		#Run the resolution function
        self.Format_Changes()			#Run the format function

    def Update_Camera(self):
        self.USB_Cam.set(cv2.CAP_PROP_EXPOSURE, Config.Exposure_Time)		#Update the exposure time
        self.USB_Cam.set(cv2.CAP_PROP_GAIN, Config.Analogue_Gain)			#Update analogue gain (brightness)
               
        self.Camera_Resolution.setText(f"<b><u>Resolution</u><br>{self.Keys[Config.Fmt_Num]}: {self.Res_Width} x {self.Res_Height} [{Config.Res_Num}]")	#Change the resolution in QLabel        
        
        self.ret, self.Frame = self.USB_Cam.read()	#Capture image(BGR Format)	#(Boolean: Detect frame captured or not, Numpy Array: The capture Image)
        
        if not self.ret or self.Frame is None:		#Wait the camera to open
            return
        
        #print(self.Frame[0,0])		#Uncomment it to see the channel value to know RGB / BGR
        self.Frame = cv2.cvtColor(self.Frame, cv2.COLOR_BGR2RGB)		#Convert Color from BGR To RGB
        #BGR2RGB = RGB2BGR Its different name but same function god damn it, it just swap the Red and Blue channel
        
        h, w, ch = self.Frame.shape		#(Image height, Image Width, Image Channels)
        bytes_per_line = ch * w		#Calculate the number of bytes per row fpr all channels
        self.QImg = QImage(self.Frame.data, w, h, bytes_per_line, QImage.Format_RGB888)	#(Pixel data,_,_,_,#RGB888 mean view first channel as R, second as G, third as B, 8 bits or 255 for each channel])
        Scaled_Image = self.QImg.scaled(self.Camera_Input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)		#Scale the image to fit the QLabel Size
        self.Camera_Input.setPixmap(QPixmap.fromImage(Scaled_Image))		#Place Image into QLabel (Continuously)

        self.Camera_Resolution.setText(f"<b><u>Resolution</u><br>{self.Keys[Config.Fmt_Num]}: {self.Res_Width} x {self.Res_Height} [{Config.Res_Num}]")	#Change the resolution in QLabel
           
        if GPIO.input(Config.PIN) == GPIO.HIGH and not Config.Sensor_Wait:		#Trigger function once if sensor is HIGH
            Config.Sensor_Wait = True			#Make sure function only run once
            threading.Thread(target=self.Trigger_Transforms).start()
        elif GPIO.input(Config.PIN) == GPIO.LOW:			#Reset if sensor is LOW
            Config.Sensor_Wait = False			#Reset
            
    def Trigger_Transforms(self):		
        h2, w2, ch2 = self.Frame.shape	#(Image height, Image Width, Image Channels)
        Scale_Factor = Config.Input_Size[0] / min(h2, w2)		#Get the smaller side	Calculate scale factor
        new_h, new_w = int(h2 * Scale_Factor), int(w2 * Scale_Factor)	#Get the new size of image based on scale factor (Keep Ratio)
        
        Resize = cv2.resize(self.Frame, (new_w, new_h), interpolation = cv2.INTER_LANCZOS4)		#Resize based on new size and use LANCZOS4 filter for better image

        Vertical_Cut = (new_h - Config.Input_Size[0]) // 2		#Calculate pixels to be remove to fit into desired size, //2 because it split up and down
        Horizontal_Cut = (new_w - Config.Input_Size[0]) //2		#Calculate pixels to be remove to fit into desired size, //2 because it split left and right
        Crop = Resize[Vertical_Cut:(Vertical_Cut + Config.Input_Size[0]), Horizontal_Cut:(Horizontal_Cut + Config.Input_Size[0])]		#Crop the image to the desired size
        #Verticle:(Vertical+Size) mean the pixel we want to keep, like 37:261, pixel 37 to 261 is the pixels to keep, other remove
        
        self.Reverse = Crop.copy()		#Copy image
        h3, w3, ch3 = self.Reverse.shape		#(Image height, Image width, Image Channels)
        bytes_per_line_3 = ch3 * w3		#Calculate the number of bytes per row fpr all channels
        self.QImg_Capture = QImage(self.Reverse.data, w3, h3, bytes_per_line_3, QImage.Format_RGB888)	#(Pixel data,_,_,_,#RGB888 mean view first channel as R, second as G, third as B, 8 bits or 255 for each channel])
      
        self.Transforms_Done.emit(self.QImg_Capture)		#Send signal back to GUI along with these data
        
        self.Save_Image()
        
    def Update_Result(self, Capture):		#Function to update the latest data to display
        self.Capture.setPixmap(QPixmap.fromImage(Capture))				#Place Image into QLabel

    def Create_Folder(self):
        self.Dataset_Folder = os.path.join(Config.Main_Path, Config.Main_Folder)		#Combine both path together
        
        for sub in Config.SubFolders:								#Create the folder one by one
           Folders = os.path.join(self.Dataset_Folder, sub)		#Combine both path together
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
            cv2.imwrite(Save_Path, cv2.cvtColor(self.Reverse, cv2.COLOR_RGB2BGR))		#Convert image to BGR Format and save it.
            #BGR2RGB = RGB2BGR Its different name but same function god damn it, it just swap the Red and Blue channel
            
            self.Path_Label.setText(f"<b>Path:</b> {Save_Path}")		#Show the saved path in this QLabel
            Config.Button_Wait = False
            
    def Exposure_Changes(self, value):		#Function that update exposure time based on slider
        Config.Exposure_Time = value		#Change to latest exposure time
        
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
    
    def Resolution_List(self):			#Function to get all the resolution from the USB Camera
        List = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{Config.Video_Index}", "--list-formats-ext"], stdout=subprocess.PIPE, text = True)
                                #Run Linux command in Python					,Print standard output to Python(instead of terminal), in String
        
        Resolution = {}		#Empty Dictionary / Bunch of lists seperated by keys
        Current_Format = None	#A variable to record the format
        
        for x in List.stdout.splitlines():		#Get every line in the List, standard output, is split into individual lines
            Format = re.search(r"\[\d+\]:\s*'(\w+)'", x)	#Search all the lines for the desired data (Stop when found one, not all)
            #Backslash(\) mean search for this specific thing, EXAMPLE: \[ mean find [ , \] mean find ] , however : no need \ because : is not special in python
            #\d mean digit(0 to 9), + mean 1 or more consecutive digit (123, 54, 15, 96821), if Line is 456EA12 then output is 456, use re.findall() to get all data
            #\s mean empty space, * mean 0 or more consecutive spaces
            #\w mean letter (a, b, c.....z), + mean 1 or more consecutive letter, () mean group it which can be extracted later
            #' ' is similar to 0, no special meaning so no need \

            if Format:	#Run if Format has something / True
                Current_Format = Format.group(1)	#Get the format from first group, leaving blank () will return whole thing
                Resolution[Current_Format] = []		#Create a empty list

            if Current_Format:		#Run if Current_Format has something / True
                Resolution_Available = re.search(r"Size:\s*Discrete\s*(\d+)x(\d+)", x)

                if Resolution_Available:	#Run Resolution_Available has something / True
                    w3, h3 = Resolution_Available.groups()	#Get the width and height from all groups, this is groups not group
                    Resolution[Current_Format].append(f"{w3}x{h3}")			#Record width and height

        return Resolution		#Produce the dictionary as output

    def Resolution_Changes(self):
        self.USB_Cam.release()		#Close camera
        if Config.Res_Num > len(self.Camera_List[self.Keys[Config.Fmt_Num]]) - 1 :	#Ensure the Res_Num dont exceed maximum number of item in the List
            Config.Res_Num = len(self.Camera_List[self.Keys[Config.Fmt_Num]]) - 1	#-1 because start with 0
        elif Config.Res_Num < 0:	#Ensure the Res_Num dont exceed minimum number of item in the List
            Config.Res_Num = 0

        if Config.Res_Num == 12:		#Check if the number is 12
            Config.Res_Num = 14			#Skip 12 because cant load / error		#Occur when testing camera
        elif Config.Res_Num == 13:		#Check if the number is 13
            Config.Res_Num = 11			#Skip 13 because cant load / error		#Occur when testing camera

        self.USB_Cam = cv2.VideoCapture(Config.Video_Index, cv2.CAP_V4L2)	#Open the camera(Index) using V4L2 driver

        self.Res_Width, self.Res_Height = map(int, self.Camera_List[self.Keys[Config.Fmt_Num]][Config.Res_Num].split('x'))	#map is a built-in function, here it apply int to every item in the list
        
        self.USB_Cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.Res_Width)		#Setup the resolution (width)
        self.USB_Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.Res_Height)		#Setup the resolution (height)
                
    def Format_Changes(self):
        if Config.Fmt_Num > len(self.Keys) - 1:		#Ensure the Fmt_Num dont exceed the maximum number of item in list
            Config.Fmt_Num = len(self.Keys) - 1		#-1 because start with 0
        elif Config.Fmt_Num < 0:					#Ensure the Fmt_Num dont exceed minimum number of item in the List
            Config.Fmt_Num = 0

        self.USB_Cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.Keys[Config.Fmt_Num]))	#Change camera format, * mean unpack string, Example: YUYV become 'Y', 'U', 'Y', 'V'
            
    def closeEvent(self, event):		#Fixed Name, Function of Qt triggerd when window is shutting down
        if self.USB_Cam and self.USB_Cam.isOpened():			#Check camera is on or not
            self.USB_Cam.release()	#Stop the camera
            
        event.accept()				#Allow Window to close
        
    def Closing(self):		#Function for ESC Button
        self.close()		#Close Window
        sys.exit(0)			#Close Entire Program and return code 0 if success
    
    def keyPressEvent(self, event):		#Fixed name, Function to detect keyboard
        if event.key() == Qt.Key_Escape:		#Detect ESC Button
            self.Closing()			#Trigger Closing() Function
        elif event.key() == Qt.Key_T:			#Detect T Button
            if not Config.Button_Wait:		#Run the content when detecting False
                Config.Button_Wait = True
                threading.Thread(target=self.Trigger_Transforms).start()	#Run the self.Trigger_Model function in background
        elif event.key() == Qt.Key_E:			#Detect E Button
            Config.Res_Num +=1					#Move to next resolution available
            self.Resolution_Changes()    		#Run the function to change resolution
        elif event.key() == Qt.Key_Q:			#Detect Q Button
            Config.Res_Num -=1					#Move to previous resolution
            self.Resolution_Changes()			#Run the function to change resolution
        elif event.key() == Qt.Key_R:			#Detect R Button
            Config.Fmt_Num +=1					#Move to next format
            self.Format_Changes()				#Run the function to change format
        elif event.key() == Qt.Key_W:			#Detect W Button
            Config.Fmt_Num -=1					#Move to previous format
            self.Format_Changes()				#Run the function to change format
        elif event.key() == Qt.Key_Z:				#Detect Z Button
            if Config.DS_Type == "train":		#Change dataset type from train to val
                Config.DS_Type = "val"
            else:
                Config.DS_Type = "train"		#Change dataset type from val to train
            self.Type_Label.setText(f"<b><u>Type</u><br>{Config.DS_Type}")	#Change Text of QLabel
            
        elif event.key() == Qt.Key_C:				#Detect C Button
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

