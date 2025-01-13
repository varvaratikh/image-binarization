import cv2
import time
from PIL import ImageFont, ImageDraw, Image
import numpy as np

font_path = '/Users/varvaratihonova/PycharmProjects/ImageBinarization/laba/laba_3/arialmt.ttf'
font_size = 32

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

prev_time = 0

font = ImageFont.truetype(font_path, font_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        eyes_open = len(eyes) > 0

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        smiling = len(smiles) > 0

        if not smiling:
            message = "Улыбнись"
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)
            draw.text((x, y - 10), message, font=font, fill=(0, 0, 255))
            frame = np.array(pil_img)

        if not eyes_open:
            message = "Открой глаза"
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)
            draw.text((x, y - 40), message, font=font, fill=(0, 0, 255))
            frame = np.array(pil_img)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 255), 2)

    curr_time = time.time()
    time_difference = curr_time - prev_time
    if time_difference > 0:
        fps = 1 / time_difference
        prev_time = curr_time

        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 30), f"FPS: {fps:.2f}", font=font, fill=(255, 255, 255))
        frame = np.array(pil_img)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
