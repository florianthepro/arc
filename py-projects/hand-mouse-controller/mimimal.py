#python -m pip install opencv-python mediapipe
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils
p = 0

while True:
    ok, img = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for hnd in res.multi_hand_landmarks:
            draw.draw_landmarks(img, hnd, mp.solutions.hands.HAND_CONNECTIONS)

    c = time.time()
    fps = 1 / (c - p) if c - p > 0 else 0
    p = c

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
