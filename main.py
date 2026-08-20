import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# 1. Initialize MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Download model if not exists
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    import urllib.request
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
landmarker = HandLandmarker.create_from_options(options)

# 2. Turn on Webcam
cap = cv2.VideoCapture(0)

# Variables to remember previous point coordinates
px, py = 0, 0

# Create an empty black canvas to draw on
canvas = None

# Set default drawing color (BGR format: Blue=255, Green=0, Red=0)
draw_color = (255, 0, 0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip camera horizontally so movement feels natural like a mirror
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Create the drawing canvas on the first run frame
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert frame to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = landmarker.detect(mp_image)

    # Check if hand landmarks are found
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw dots and lines on the hand image
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Get coordinates for Index Tip (Landmark 8) & Middle Tip (Landmark 12)
            # Coordinates are normalized (0 to 1), so multiply by screen height/width
            index_x = int(hand_landmarks[8].x * w)
            index_y = int(hand_landmarks[8].y * h)
            
            index_pip_y = int(hand_landmarks[6].y * h)
            middle_y = int(hand_landmarks[12].y * h)
            middle_pip_y = int(hand_landmarks[10].y * h)

            # Check finger states (UP if Tip Y coordinate is LESS than Joint PIP Y coordinate)
            index_up = index_y < index_pip_y
            middle_up = middle_y < middle_pip_y

            # GESTURE LOGIC:
            # 1. Selection Mode (Index AND Middle up) -> Stop drawing, move cursor
            if index_up and middle_up:
                px, py = 0, 0  # Reset previous point
                cv2.circle(frame, (index_x, index_y), 15, (0, 255, 255), cv2.FILLED)

            # 2. Drawing Mode (ONLY Index up) -> Draw line from previous point to current point
            elif index_up and not middle_up:
                cv2.circle(frame, (index_x, index_y), 8, draw_color, cv2.FILLED)
                
                if px == 0 and py == 0:
                    px, py = index_x, index_y

                # Draw a line on the canvas layer
                cv2.line(canvas, (px, py), (index_x, index_y), draw_color, 7)
                px, py = index_x, index_y

            else:
                px, py = 0, 0

    # Draw Color Selection UI Boxes on top of screen
    cv2.rectangle(frame, (20, 10), (100, 60), (255, 0, 0), -1)   # Blue Box
    cv2.rectangle(frame, (120, 10), (200, 60), (0, 255, 0), -1)  # Green Box
    cv2.rectangle(frame, (220, 10), (300, 60), (0, 0, 255), -1)  # Red Box
    cv2.rectangle(frame, (320, 10), (420, 60), (0, 0, 0), -1)    # Eraser/Clear Box
    cv2.putText(frame, "CLEAR", (330, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Color Selection logic (When finger touches target area)
    if detection_result.hand_landmarks:
        if index_y < 60:
            if 20 < index_x < 100:
                draw_color = (255, 0, 0) # Blue
            elif 120 < index_x < 200:
                draw_color = (0, 255, 0) # Green
            elif 220 < index_x < 300:
                draw_color = (0, 0, 255) # Red
            elif 320 < index_x < 420:
                canvas = np.zeros((h, w, 3), dtype=np.uint8) # Clear Canvas

    # Combine drawing canvas with live video frame
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask_inv = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    
    frame = cv2.bitwise_and(frame, mask_inv)
    frame = cv2.bitwise_or(frame, canvas)

    # Display video output window
    cv2.imshow("Air Canvas - AI Hand Drawing", frame)

    # Press 'q' key on keyboard to close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()