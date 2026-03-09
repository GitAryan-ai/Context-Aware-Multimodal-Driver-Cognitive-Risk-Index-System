import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import winsound
import os
import time

# ====================== MEDIAPIPE SETUP ======================
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = FaceLandmarker.create_from_options(options)

# ====================== LANDMARK INDICES ======================
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 291, 0, 17, 37, 267]

# ====================== PARAMETERS ======================
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15

MOUTH_AR_THRESH = 0.70
MOUTH_AR_CONSEC_FRAMES = 20

RISK_YAWN = 50
RISK_DROWSY = 100

OCCLUSION_CONSEC_FRAMES = 3

MAR_SMOOTH_WINDOW = 5

ALARM_SOUND_FILE = "alarm.wav"  # optional .wav file

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

eye_closed_counter = 0
yawn_counter = 0
occlusion_counter = 0
drowsiness_status = "Alert"
alert_triggered = False
alarm_active = False
risk_score = 0
acknowledge_pressed = False

mar_history = []
frame_count = 0

last_alarm_time = 0
ALARM_INTERVAL = 1.2

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    vert1 = np.linalg.norm(mouth[2] - mouth[3])
    vert2 = np.linalg.norm(mouth[4] - mouth[5])
    horiz = np.linalg.norm(mouth[0] - mouth[1])
    mar = (vert1 + vert2) / (2.0 * horiz)
    return mar

def play_alarm():
    global last_alarm_time
    current_time = time.time()
    if current_time - last_alarm_time >= ALARM_INTERVAL:
        if os.path.exists(ALARM_SOUND_FILE):
            winsound.PlaySound(ALARM_SOUND_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            winsound.Beep(2800, 700)
            time.sleep(0.1)
            winsound.Beep(1800, 500)
        last_alarm_time = current_time

print("Drowsiness + Occlusion Detection – Camera stays open until 'Q/q' pressed")
print("Press 'Q'/'q' to quit | Press 'A' to acknowledge alarm")

cv2.namedWindow("Driver Drowsiness Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Driver Drowsiness Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam disconnected – stopping.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms < 0:
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    face_detected_now = bool(detection_result.face_landmarks)

    # Occlusion detection
    if not face_detected_now:
        occlusion_counter += 1
    else:
        occlusion_counter = 0

    is_occluded = occlusion_counter >= OCCLUSION_CONSEC_FRAMES

    face_detected = False
    x_min = y_min = x_max = y_max = 0

    eye_drowsy = False
    yawn_drowsy = False

    if face_detected_now:
        face_detected = True
        face_landmarks = detection_result.face_landmarks[0]
        h, w, _ = frame.shape
        landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks])

        left_eye = landmarks[LEFT_EYE_INDICES]
        right_eye = landmarks[RIGHT_EYE_INDICES]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        mouth = landmarks[MOUTH_INDICES]
        mar_raw = mouth_aspect_ratio(mouth)

        mar_history.append(mar_raw)
        if len(mar_history) > MAR_SMOOTH_WINDOW:
            mar_history.pop(0)
        mar = np.mean(mar_history)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Smoothed MAR: {mar:.3f}")

        for pt in left_eye.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)
        for pt in right_eye.astype(int):
            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        is_yawning = mar > MOUTH_AR_THRESH
        mouth_color = (0, 0, 255) if is_yawning else (100, 100, 100)
        for pt in mouth.astype(int):
            cv2.circle(frame, tuple(pt), 3, mouth_color, -1)

        x_min, y_min = landmarks.min(axis=0).astype(int) - 10
        x_max, y_max = landmarks.max(axis=0).astype(int) + 10
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        if ear < EYE_AR_THRESH:
            eye_closed_counter += 1
            if eye_closed_counter >= EYE_AR_CONSEC_FRAMES:
                eye_drowsy = True
        else:
            eye_closed_counter = 0

        if mar > MOUTH_AR_THRESH:
            yawn_counter += 1
            if yawn_counter >= MOUTH_AR_CONSEC_FRAMES:
                yawn_drowsy = True
        else:
            yawn_counter = 0

        risk_score = 0
        if eye_closed_counter > 5: risk_score += 40
        if yawn_counter > 10 and mar > 0.65: risk_score += 30
        if eye_drowsy: risk_score = RISK_DROWSY
        elif yawn_drowsy: risk_score = RISK_YAWN

        is_drowsy = eye_drowsy or yawn_drowsy

        if is_drowsy:
            drowsiness_status = "DROWSY!"
            alarm_active = True
            if not alert_triggered:
                alert_triggered = True
        else:
            drowsiness_status = "Alert"
            alert_triggered = False
            if alarm_active and not acknowledge_pressed:
                alarm_active = False

    else:
        drowsiness_status = "Alert"
        risk_score = 0
        alert_triggered = False

    # ALARM LOGIC
    trigger_alarm = (eye_drowsy or yawn_drowsy) or is_occluded

    if trigger_alarm:
        alarm_active = True
        play_alarm()

        flash_color = (0, 0, 255) if int(time.time() * 5) % 2 == 0 else (255, 255, 0)
        
        if is_occluded:
            msg = "FACE BLOCKED!"
            sub_msg = "Hand/object detected – Press A"
        else:
            msg = "WAKE UP!"
            sub_msg = "Drowsiness detected – Press A"

        # Slightly lower position
        text_y_offset = frame.shape[0] // 3 + 140   # ← adjusted lower here
        text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 5)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = text_y_offset

        cv2.putText(frame, msg, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, flash_color, 5)

        sub_size = cv2.getTextSize(sub_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)[0]
        sub_x = (frame.shape[1] - sub_size[0]) // 2
        cv2.putText(frame, sub_msg, (sub_x, text_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

    # KEY HANDLING
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("Quit pressed – exiting.")
        break
    if key == ord('a') or key == ord('A'):
        print("Alarm acknowledged and stopped.")
        alarm_active = False

    # ────────────────────────────────────────────────────────────────
    # TOP-LEFT STATUS BOX
    # ────────────────────────────────────────────────────────────────

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (10, 8, 25), -1)
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    if face_detected:
        cv2.rectangle(frame, (x_min-25, y_min-35), (x_max+25, y_max+45), (0, 140, 80), 8)
        cv2.rectangle(frame, (x_min-12, y_min-22), (x_max+12, y_max+32), (0, 255, 160), 2)

    panel_w, panel_h = 240, 180
    panel_x = 20
    panel_y = 20   # Top-left

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 40), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 220, 255), 3)

    status_text = f"{drowsiness_status.upper()}"
    status_color = (100, 255, 140) if drowsiness_status == "Alert" else (255, 80, 80)
    cv2.putText(frame, status_text, (panel_x + 20, panel_y + 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0), 6)
    cv2.putText(frame, status_text, (panel_x + 17, panel_y + 47),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, status_color, 4)

    risk_text = f"RISK: {risk_score}%"
    cv2.putText(frame, risk_text, (panel_x + 20, panel_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,220,255), 2)

    bar_x = panel_x + 25
    bar_y_eye = panel_y + 115
    bar_y_yawn = panel_y + 145
    bar_width = panel_w - 50
    bar_height = 10

    eye_progress = min(eye_closed_counter / EYE_AR_CONSEC_FRAMES, 1.0)
    for i in range(bar_width):
        r = int(255 * i / bar_width)
        g = int(180 * (1 - i / bar_width))
        b = 0
        color = (b, g, r)
        cv2.line(frame, (bar_x + i, bar_y_eye), (bar_x + i, bar_y_eye + bar_height),
                 color if i < int(bar_width * eye_progress) else (60,60,60), 1)

    cv2.putText(frame, "Eye", (bar_x - 15, bar_y_eye + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,255), 1)

    yawn_progress = min(yawn_counter / MOUTH_AR_CONSEC_FRAMES, 1.0)
    for i in range(bar_width):
        r = int(255 * i / bar_width)
        g = int(120 * (1 - i / bar_width))
        b = 0
        color = (b, g, r)
        cv2.line(frame, (bar_x + i, bar_y_yawn), (bar_x + i, bar_y_yawn + bar_height),
                 color if i < int(bar_width * yawn_progress) else (60,60,60), 1)

    cv2.putText(frame, "Mouth", (bar_x - 25, bar_y_yawn + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,255), 1)

    light_x = frame.shape[1] - 60
    light_y = 40
    light_color = (60, 255, 80) if drowsiness_status == "Alert" else (0, 60, 255)
    cv2.circle(frame, (light_x, light_y), 22, light_color, -1)
    cv2.circle(frame, (light_x, light_y), 26, (220, 220, 240), 2)

    if drowsiness_status != "Alert":
        tri_pts = np.array([
            [light_x - 12, light_y + 35],
            [light_x + 12, light_y + 35],
            [light_x, light_y + 60]
        ], np.int32)
        cv2.fillPoly(frame, [tri_pts], (0, 0, 255))
        cv2.putText(frame, "!", (light_x - 8, light_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    if alarm_active:
        border_color = (0, 0, 255) if int(time.time() * 6) % 2 == 0 else (255, 100, 100)
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), border_color, 8)

    cv2.imshow("Driver Drowsiness Detection", frame)

cap.release()
cv2.destroyAllWindows()