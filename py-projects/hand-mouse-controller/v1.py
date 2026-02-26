#python pip install lib
import cv2
import mediapipe as mp
import time
# Webcam öffnen
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

pTime = 0  # previous time

while True:
    success, img = cap.read()
    if not success:
        print("Kamera-Bild konnte nicht gelesen werden.")
        break

    # Bild in RGB umwandeln (für MediaPipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Hände auslesen
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Landmarks in der Konsole ausgeben
                # print(id, cx, cy)

                # Beispiel: Handgelenk (id 0) hervorheben
                if id == 0:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            # Hand-Landmarks & Verbindungen zeichnen
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # FPS berechnen
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    # FPS anzeigen
    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Bild anzeigen
    cv2.imshow("Image", img)

    # Mit 'q' abbrechen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()

