import cv2
import numpy as np
from scipy.spatial import distance
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp
import time

font_path = '/Users/varvaratihonova/PycharmProjects/ImageBinarization/laba/laba_3/arialmt.ttf'
font_size = 32

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

font = ImageFont.truetype(font_path, font_size)

def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)

frame_count = 0
prev_frame_time = time.time()
total_fps = 0
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = [
                (face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0])
                for i in [362, 385, 387, 263, 373, 380]
            ]
            right_eye = [
                (face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0])
                for i in [33, 160, 158, 133, 153, 144]
            ]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            if left_ear < EAR_THRESHOLD or right_ear < EAR_THRESHOLD:
                frame_count += 1
                if frame_count >= CONSEC_FRAMES:
                    pil_img = Image.fromarray(frame)
                    draw = ImageDraw.Draw(pil_img)
                    draw.text((50, 50), "Открой глаза", font=font, fill=(0, 0, 255))
                    frame = np.array(pil_img)
            else:
                frame_count = 0

            left_eye_np = np.array(left_eye, dtype=np.int32)
            right_eye_np = np.array(right_eye, dtype=np.int32)
            cv2.polylines(frame, [left_eye_np], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.polylines(frame, [right_eye_np], isClosed=True, color=(0, 255, 0), thickness=1)

    current_frame_time = time.time()
    time_difference = current_frame_time - prev_frame_time
    fps = 1 / time_difference
    prev_frame_time = current_frame_time

    total_fps += fps
    frame_counter += 1

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face and Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

average_fps = total_fps / frame_counter if frame_counter > 0 else 0
print(f"Средний FPS: {average_fps:.2f}")
