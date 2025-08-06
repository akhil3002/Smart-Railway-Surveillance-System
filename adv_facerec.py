import cv2
from deepface import DeepFace
import requests

# Webcam config
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load target image only once
target_img = cv2.imread("target.jpg")

frame_skip = 3  # Process every 3rd frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip == 0:
        try:
            # Perform face verification directly on arrays
            result = DeepFace.verify(
                img1_path=frame,
                img2_path=target_img,
                model_name="SFace",  # Much faster
                enforce_detection=False
            )

            if result.get("verified"):
                print("✅ Match Found - Sending Alert...")
                try:
                    response = requests.get("http://127.0.0.1:8000/target_face_alert")
                    print("✅ Alert Sent:", response.json())
                except Exception as e:
                    print("❌ Alert send failed:", e)

        except Exception as e:
            print("❌ Face verification error:", e)

    # Show the frame
    cv2.imshow("Live Face Match", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()