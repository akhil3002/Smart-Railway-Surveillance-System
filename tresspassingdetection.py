import cv2
import numpy as np
import requests
from ultralytics import YOLO

# Load the models
track_model = YOLO("runs\\segment\\track_segmentation_1\\weights\\best.pt")# Segmentation model for railway track
person_model = YOLO("yolov8n.pt")     # Detection model for person

# Start video capture
cap = cv2.VideoCapture("tracktestvideo.mp4")  #path to video file

alert_sent = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Run segmentation model on frame
    track_results = track_model(frame)[0]
    track_mask = None

    # Create mask from segmentation output
    if track_results.masks:
        masks = track_results.masks.data.cpu().numpy()
        combined_mask = np.any(masks > 0.5, axis=0).astype(np.uint8)  # Combine all masks
        mask_resized = cv2.resize(combined_mask, (width, height))
        track_mask = mask_resized

        # Optional: visualize combined mask
        colored_mask = np.zeros_like(frame)
        colored_mask[track_mask == 1] = (0, 0, 255)
        frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

    # Run person detection
    person_results = person_model(frame)[0]
    person_detected_on_track = False

    for box in person_results.boxes:
        cls = int(box.cls[0])
        if person_model.names[cls] == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point

            # Draw person detection (optional)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if center of person is in track region
            if track_mask is not None and track_mask[cy, cx] == 1:
                person_detected_on_track = True

    # Alert logic
    if person_detected_on_track and not alert_sent:
        print("🚨 Person detected on railway track! Sending alert...")
        try:
            res = requests.get("http://127.0.0.1:8000/track_alert")
            print("✅ Alert sent! Response:", res.json())
            alert_sent = True
        except Exception as e:
            print("❌ Failed to send alert:", e)
    elif not person_detected_on_track:
        alert_sent = False  # Reset alert flag when no person on track

    # Display the frame
    cv2.imshow("Smart Railway Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
