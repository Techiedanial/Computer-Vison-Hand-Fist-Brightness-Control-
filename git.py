# Importing Libraries 
import cv2 
import mediapipe as mp 
import screen_brightness_control as sbc 
import numpy as np 

# Initializing the Model 
mpHands = mp.solutions.hands 
hands = mpHands.Hands(
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=1  # We will track only one hand
) 

Draw = mp.solutions.drawing_utils 

# Start capturing video from webcam 
cap = cv2.VideoCapture(0) 

# Function to detect if the hand is open or closed
def is_hand_closed(landmarks, height, width):
    """
    Function to detect if the hand is closed (fist) or open.
    Returns True if the hand is closed, False otherwise.
    """
    # Get coordinates of the tips of the fingers
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Get coordinates of the MCP (metacarpophalangeal) joints of the fingers
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    # Calculate the distances between the tips of the fingers and their MCP joints
    index_finger_folded = index_tip.y > index_mcp.y  # Folded if tip is below MCP
    middle_finger_folded = middle_tip.y > middle_mcp.y
    ring_finger_folded = ring_tip.y > ring_mcp.y
    pinky_finger_folded = pinky_tip.y > pinky_mcp.y

    # If all fingers except the thumb are folded, hand is considered closed (fist)
    if index_finger_folded and middle_finger_folded and ring_finger_folded and pinky_finger_folded:
        return True
    return False

# Current brightness level
current_brightness = sbc.get_brightness(display=0)[0]

while True: 
    # Read video frame by frame 
    _, frame = cap.read() 

    # Flip image 
    frame = cv2.flip(frame, 1) 

    # Convert BGR image to RGB image 
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    # Process the RGB image 
    Process = hands.process(frameRGB) 

    landmarkList = [] 
    # if hands are present in image(frame) 
    if Process.multi_hand_landmarks: 
        # Detect landmarks of the right hand (only one hand in this case)
        for handlm in Process.multi_hand_landmarks: 
            for _id, landmarks in enumerate(handlm.landmark): 
                # store height and width of image 
                height, width, color_channels = frame.shape 

                # calculate and append x, y coordinates 
                # of handmarks from image(frame) to lmList 
                x, y = int(landmarks.x*width), int(landmarks.y*height) 
                landmarkList.append(landmarks)  # store normalized coordinates directly

            # Draw the hand landmarks
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

        # If landmarks list is not empty
        if landmarkList != []:
            # Check if the hand is open or closed (fist)
            if is_hand_closed(landmarkList, height, width):
                # Hand is closed, reduce brightness
                current_brightness = max(0, current_brightness - 5)
                sbc.set_brightness(current_brightness)
            else:
                # Hand is open, increase brightness
                current_brightness = min(100, current_brightness + 5)
                sbc.set_brightness(current_brightness)
    
    # Display Video and when 'q' is entered, destroy the window 
    cv2.imshow('Image', frame) 
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()
