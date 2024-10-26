# Computer-Vison-Hand-Fist-Brightness-Control
This project utilizes computer vision to create a program that adjusts screen brightness based on hand gestures captured via webcam.

# Description
This program leverages the MediaPipe library to detect hand landmarks in real-time.It then analyzes the hand posture to determine if it's open or closed (fist).Based on the detected hand gesture:


# Closed Fist: Reduces screen brightness by 5 units.

Open Hand: Increases screen brightness by 5 units.

The program maintains a minimum brightness level of 0 and a maximum of 100 to prevent excessive adjustments.

# Technology Stack
Programming Language: Python
Libraries:
OpenCV (cv2)
MediaPipe (mp)
screen_brightness_control (sbc)
NumPy (np)

