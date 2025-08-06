import cv2
import requests
import time
from ultralytics import YOLO

# Load the trained model
model = YOLO("C:\\Users\\akhil\\Desktop\\Smart Railway Surveillance System\\runs\\detect\\train\\weights\\best.pt")

# Start capturing from the webcam
cap = cv2.VideoCapture(r"R:\UCF CRIME\train\Shooting\Shooting054_x264.mp4")
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

alert_sent = False  # Flag to track if alert has been sent
alert_cooldown = 10  # Cooldown time in seconds before sending another alert
last_alert_time = 0  # Timestamp of last alert

while True:
    ret, img = cap.read()
    results = model(img, stream=True, conf=0.50)

    weapon_detected = False  # Flag to check if weapon is detected in the frame

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            cls = int(box.cls[0])

            label = model.names[cls]  # Get the class name
            print(f"Detected: {label} (Confidence: {confidence:.2f})")

            if label.lower() == "weapon":
                weapon_detected = True  # Set flag if weapon detected

                # Check if alert was already sent and cooldown time passed
                if not alert_sent or (time.time() - last_alert_time > alert_cooldown):
                    print("🚨 Weapon detected! Sending alert...")

                    try:
                        response = requests.get("http://127.0.0.1:8000/weapon_alert")
                        print("✅ Alert Sent! Response:", response.json())
                        alert_sent = True  # Set alert flag
                        last_alert_time = time.time()  # Update last alert time
                    except requests.exceptions.RequestException as e:
                        print("❌ Error sending alert:", e)

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Reset alert flag if no weapon is detected
    if not weapon_detected:
        alert_sent = False

    cv2.imshow('Real-Time Weapon Detection', img)

    if cv2.waitKey(1) == ord('q'):  # Exit on 'q' key
        break

cap.release()
cv2.destroyAllWindows()
