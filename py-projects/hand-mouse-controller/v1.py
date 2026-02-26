import importlib, sys

importlib.import_module("cv2") if importlib.util.find_spec(
    "cv2"
) else importlib.import_module("pip").main(["install", "opencv-python"])
import cv2
import importlib, sys

importlib.import_module("mediapipe") if importlib.util.find_spec(
    "mediapipe"
) else importlib.import_module("pip").main(["install", "mediapipe"])
import mediapipe as mp
import time

# Webcam öffnen
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils
p_time = 0  # previous time
while True:
    success, img = cap.read()
    if not success:
        print("Kamera-Bild konnte nicht gelesen werden.")
        break
    # BGR → RGB für MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            h, w, c = img.shape
            for idx, lm in enumerate(hand_lms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Handgelenk (Landmark 0) markieren
                if idx == 0:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
    # FPS berechnen
    c_time = time.time()
    fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
    p_time = c_time
    # FPS anzeigen
    cv2.putText(
        img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
    )
    # Bild anzeigen
    cv2.imshow("Image", img)
    # Mit 'q' abbrechen
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
