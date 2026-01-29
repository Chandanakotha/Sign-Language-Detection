import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

# ---------------- Hand Detector Class ----------------
class HandDetector:
    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        try:
            with open('hand_landmarker.task', 'rb') as f:
                model_content = f.read()
            base_options = python.BaseOptions(model_asset_buffer=model_content)
        except FileNotFoundError:
            # Fallback if the model file isn't found
            base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_hand_presence_confidence=detectionCon,
            min_tracking_confidence=minTrackCon,
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=True, flipType=True):
        if img is None:
            return [], None

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        detection_result = self.landmarker.detect(mp_image)

        allHands = []
        h, w, c = img.shape

        if detection_result.hand_landmarks:
            for i, landmarks in enumerate(detection_result.hand_landmarks):
                myHand = {}
                mylmList = []
                xList, yList = [], []

                for lm in landmarks:
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                # Handedness
                if len(detection_result.handedness) > i:
                    label = detection_result.handedness[i][0].category_name
                else:
                    label = "Right"

                if flipType:
                    myHand["type"] = "Left" if label == "Right" else "Right"
                else:
                    myHand["type"] = label

                allHands.append(myHand)

        return allHands, img

# ---------------- Video Loop ----------------
def video_loop():
    cap = cv2.VideoCapture(0)  # 0 = default camera
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    hd = HandDetector(maxHands=2)
    prevTime = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: Empty frame received, skipping...")
            continue

        hands, img = hd.findHands(frame, draw=False)

        # Debug logging similar to your previous logs
        for i, hand in enumerate(hands):
            print(f"{time.time():.0f}  ch{i}=" + "+"*15, hand.get("center", (0,0)), hand.get("type", "?"))

        # FPS display
        currTime = time.time()
        fps = 1 / (currTime - prevTime) if prevTime else 0
        prevTime = currTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Show camera window
        cv2.imshow("Sign Language Detection", img)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- Main ----------------
if __name__ == "__main__":
    video_loop()
