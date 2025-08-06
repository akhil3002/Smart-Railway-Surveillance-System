import cv2
import requests
import time
import os
import concurrent.futures
from deepface import DeepFace
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load Haar cascade for face detection 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

last_alert_time = 0
cooldown = 5
frame_count = 0
process_every_n_frames = 5

def process_face(face_img):
    try:
        # Find matches in the database for the detected face
        result = DeepFace.find(img_path=face_img, 
                             db_path="face_data/", 
                             model_name="SFace",
                             enforce_detection=False)  # Avoid re-detection
        return result
    except Exception as e:
        print(f"DeepFace Error: {e}")
        return None

with concurrent.futures.ThreadPoolExecutor() as executor:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        display_frame = frame.copy()
        
        # Detect faces first
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if frame_count % process_every_n_frames == 0 and len(faces) > 0:
            matched_names = []
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face ROI (Region of Interest)
                face_roi = frame[y:y+h, x:x+w]
                
                # Submit face for recognition
                future = executor.submit(process_face, face_roi)
                result = future.result()

                if result and any(len(res) > 0 for res in result):
                    for res in result:
                        if len(res) > 0:
                            # Get the first match
                            identity = os.path.splitext(os.path.basename(res.iloc[0]["identity"]))[0]
                            matched_names.append(identity)
                            
                            # Draw rectangle and label
                            cv2.rectangle(display_frame, 
                                        (x, y), 
                                        (x + w, y + h), 
                                        (0, 255, 0),  # Green
                                        2)
                            cv2.putText(display_frame, 
                                      identity, 
                                      (x, y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.9, 
                                      (0, 255, 0), 
                                      2)

            if matched_names:
                current_time = time.time()
                if current_time - last_alert_time > cooldown:
                    alert_data = {"names": matched_names}
                    response = requests.post("http://127.0.0.1:8000/alert", json=alert_data)
                    print(f"Alert Sent: {response.status_code}, {response.text}")
                    last_alert_time = current_time

        frame_count += 1
        cv2.imshow("Live Face Recognition", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()