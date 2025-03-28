from fastapi import FastAPI, Form, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from asltohuman import convert_asl_to_human_text, generate_message_from_gestures
import cv2
import os
import cv2 as cv
from app import draw_bounding_rect, draw_info, draw_info_text, draw_landmarks,draw_point_history, get_args, select_mode, calc_bounding_rect, calc_landmark_list,convert_asl_to_human_text,pre_process_landmark,pre_process_point_history, logging_csv
import mediapipe as mp
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

app = FastAPI()

args = get_args()
global camera_active, current_cap
camera_active = True

cap_device = args.device
cap_width = args.width
cap_height = args.height

use_static_image_mode = args.use_static_image_mode
min_detection_confidence = args.min_detection_confidence # Threshold for detecting hands 
min_tracking_confidence = args.min_tracking_confidence # Threshold for tracking hands 

use_brect = True # Draw a bounding rectangle around the hand 

# Camera preparation ###############################################################
cap = cv.VideoCapture(cap_device)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")
else:
    print("Camera opened successfully.")
if not cap.isOpened():
    raise RuntimeError("Could not open camera")
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

 # Model load 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=use_static_image_mode,
    max_num_hands=2,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
) # Mediapipeline pretrained models for tracking hands 

keypoint_classifier = KeyPointClassifier() # Used for recognizing static hand gestures using hand landmarks 
point_history_classifier = PointHistoryClassifier()
 # Used for recognizing dynamic hand gestures Tracks the movement pattern of the hand over time and uses time sequenceds points or past hand positions to recognize gestures 

# Read labels 
with open('hand-gesture-recognition-mediapipe/model/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]
with open(
        'hand-gesture-recognition-mediapipe/model/point_history_classifier/point_history_classifier_label.csv',
        encoding='utf-8-sig') as f:
    point_history_classifier_labels = csv.reader(f)
    point_history_classifier_labels = [
        row[0] for row in point_history_classifier_labels
    ]

# FPS Measurement 
cvFpsCalc = CvFpsCalc(buffer_len=10) # measures FPS, stores FPS values of the last 10 frames to smooth out fluctuations 

# Coordinate history 
history_length = 16
point_history = deque(maxlen=history_length) # stores the movement history (coordinates) of a specific hand landmark, last 16 frames   
finger_gesture_history = deque(maxlen=history_length) # keeping track of the last 16 frame's classified gestures 

recognized_gestures = []  # List to store the recognized gestures from the video input
last_gesture = None  # Variable to track the last recognized gesture

def generate():
    message = None  # Initialize message
    while camera_active == True:
        fps = cvFpsCalc.get()

        # Camera capture
        ret, image = cap.read()  # if frame was successfully read ret = True , image - that particular frame 
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False  # Improves performance by preventing unnecessary memory copies before processing
        results = hands.process(image)  # to detect hand landmarks using mediapipe 
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None: 
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id != "Not applicable":
                    current_gesture = keypoint_classifier_labels[hand_sign_id]
                    if current_gesture != last_gesture:  # Check if the gesture has changed
                        recognized_gestures.append(current_gesture)  # Append the new gesture
                        last_gesture = current_gesture  # Update the last recognized gesture
                        print(recognized_gestures)  # Print the recognized gestures

                else:
                    last_gesture = None  # Reset if no gesture is recognized

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(   
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            last_gesture = None  # Reset if no hands are detected

        debug_image = draw_point_history(debug_image, point_history)

        # Encode and yield frame
        _, buffer = cv2.imencode('.jpg', debug_image)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    # After stopping, generate the message from recognized gestures
    if recognized_gestures:
        message = generate_message_from_gestures(recognized_gestures)
        print("Generated Message:", message)  # You can also return this to the frontend if needed
    else:
        message = "No gestures recognized."  # Default message if no gestures

    return message

# Set up Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@app.get("/", response_class = HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/convert_asl")
async def convert_asl(asl_sentence: str = Form(...)):
    human_text = convert_asl_to_human_text(asl_sentence)
    return {"converted_text": human_text}

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/stop_video_feed")
async def stop_video_feed():
    global camera_active
    camera_active = False
    return {"status": "Video feed stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
