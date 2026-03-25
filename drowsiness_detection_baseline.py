import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import winsound
import os
import time
from collections import deque
import sys

# ====================== SIMPLE SETUP ======================
print("Starting Driver Monitoring System...")

# Check if model exists
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("\n" + "="*60)
    print("ERROR: face_landmarker.task not found!")
    print("="*60)
    print("Please download from:")
    print("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    print(f"\nSave it in: {os.path.dirname(__file__)}")
    print("="*60)
    input("Press Enter to exit...")
    sys.exit(1)

# Initialize MediaPipe
try:
    BaseOptions = python.BaseOptions
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = FaceLandmarker.create_from_options(options)
    print("✓ MediaPipe model loaded")
except Exception as e:
    print(f"Error loading model: {e}")
    input("Press Enter to exit...")
    sys.exit(1)

# ====================== LANDMARK INDICES ======================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 0, 17, 37, 267]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# ====================== THRESHOLDS ======================
# Base thresholds
EYE_THRESH = 0.20
BLINK_THRESH = 0.5
DANGEROUS_CLOSURE_THRESH = 2.0
MOUTH_THRESH = 0.70
YAWN_THRESH = 2.5
HEAD_TURN_THRESH = 0.30

# Timer thresholds (seconds)
LOOK_LEFT_ALERT_TIME = 3.0
LOOK_RIGHT_ALERT_TIME = 3.0
LOOK_DOWN_ALERT_TIME = 4.0
NO_FACE_ALERT_TIME = 3.0  # Faster alert for face missing

# Night mode adjustments
NIGHT_EYE_THRESH = 0.18
NIGHT_LOOK_LEFT_TIME = 3.5
NIGHT_LOOK_RIGHT_TIME = 3.5
NIGHT_LOOK_DOWN_TIME = 5.0
NIGHT_NO_FACE_TIME = 4.0

# Other thresholds
DROWSY_CONSECUTIVE_FRAMES = 20
ALARM_COOLDOWN = 3
FRAME_SKIP = 2

# ====================== WARNING TYPES ======================
WARNING_TYPES = {
    'DROWSY': {'message': 'DROWSY DETECTED!', 'color': (0, 0, 255), 'sound': 'beep', 'priority': 5},
    'YAWNING': {'message': 'EXCESSIVE YAWNING!', 'color': (0, 165, 255), 'sound': 'beep', 'priority': 4},
    'LOOKING_LEFT': {'message': 'LOOKING LEFT TOO LONG!', 'color': (255, 255, 0), 'sound': 'beep', 'priority': 3},
    'LOOKING_RIGHT': {'message': 'LOOKING RIGHT TOO LONG!', 'color': (255, 255, 0), 'sound': 'beep', 'priority': 3},
    'PHONE': {'message': 'PHONE DETECTED!', 'color': (255, 0, 255), 'sound': 'double_beep', 'priority': 4},
    'NO_FACE': {'message': 'FACE NOT VISIBLE!', 'color': (128, 128, 128), 'sound': 'beep', 'priority': 2},
    'MICROSLEEP': {'message': 'MICROSLEEP DETECTED!', 'color': (0, 0, 255), 'sound': 'continuous', 'priority': 5}
}

# ====================== SOUND SYSTEM ======================
class SoundAlert:
    def __init__(self):
        self.last_beep_time = 0
        self.last_face_alert_time = 0
        self.beep_cooldown = 0.8
        self.face_alert_cooldown = 0.4
        self.use_winsound = True
        self.test_sound()
    
    def test_sound(self):
        try:
            winsound.Beep(1000, 100)
            print("✓ Sound system ready")
            self.use_winsound = True
        except:
            print("⚠️ winsound not available, using console beep")
            self.use_winsound = False
    
    def beep(self, frequency=1000, duration=300, is_face_alert=False):
        current_time = time.time()
        
        if is_face_alert:
            if current_time - self.last_face_alert_time < self.face_alert_cooldown:
                return
            self.last_face_alert_time = current_time
        else:
            if current_time - self.last_beep_time < self.beep_cooldown:
                return
            self.last_beep_time = current_time
        
        try:
            if self.use_winsound:
                winsound.Beep(frequency, duration)
            else:
                print('\a', end='', flush=True)
                sys.stdout.write('\x07')
                sys.stdout.flush()
        except:
            pass
    
    def double_beep(self):
        self.beep(1000, 200)
        time.sleep(0.1)
        self.beep(1000, 200)
    
    def continuous_beep(self, count=3):
        for i in range(count):
            self.beep(2000, 300)
            time.sleep(0.05)
    
    def fast_face_alert(self):
        for i in range(5):
            self.beep(1500, 150, is_face_alert=True)
            time.sleep(0.15)

# ====================== LIGHT ANALYZER ======================
class LightAnalyzer:
    def __init__(self):
        self.brightness_history = deque(maxlen=15)
        self.is_night_mode = False
        
    def analyze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        self.brightness_history.append(avg_brightness)
        smoothed = np.mean(self.brightness_history)
        
        if smoothed < 60:
            self.is_night_mode = True
        elif smoothed > 90:
            self.is_night_mode = False
        
        return self.is_night_mode, smoothed

# ====================== TIMER-BASED DETECTOR ======================
class TimerBasedDetector:
    def __init__(self):
        self.looking_left_start = None
        self.looking_right_start = None
        self.looking_down_start = None
        self.current_look_direction = "CENTER"
        
    def update_head_direction(self, head_turn, looking_down, current_time, is_night):
        """Update timers based on head direction with night mode adjustments"""
        # Get night-adjusted thresholds
        left_thresh = NIGHT_LOOK_LEFT_TIME if is_night else LOOK_LEFT_ALERT_TIME
        right_thresh = NIGHT_LOOK_RIGHT_TIME if is_night else LOOK_RIGHT_ALERT_TIME
        down_thresh = NIGHT_LOOK_DOWN_TIME if is_night else LOOK_DOWN_ALERT_TIME
        
        # Determine current look direction
        if looking_down:
            direction = "DOWN"
        elif head_turn > HEAD_TURN_THRESH:
            direction = "RIGHT"
        elif head_turn < -HEAD_TURN_THRESH:
            direction = "LEFT"
        else:
            direction = "CENTER"
        
        # Reset timers when direction changes
        if direction != self.current_look_direction:
            self.looking_left_start = None
            self.looking_right_start = None
            self.looking_down_start = None
            self.current_look_direction = direction
        
        warnings = []
        duration = 0
        
        # Start or update timers based on current direction
        if direction == "LEFT":
            if self.looking_left_start is None:
                self.looking_left_start = current_time
            else:
                duration = current_time - self.looking_left_start
                if duration > left_thresh:
                    warnings.append(('LOOKING_LEFT', duration))
        elif direction == "RIGHT":
            if self.looking_right_start is None:
                self.looking_right_start = current_time
            else:
                duration = current_time - self.looking_right_start
                if duration > right_thresh:
                    warnings.append(('LOOKING_RIGHT', duration))
        elif direction == "DOWN":
            if self.looking_down_start is None:
                self.looking_down_start = current_time
            else:
                duration = current_time - self.looking_down_start
                if duration > down_thresh:
                    warnings.append(('PHONE', duration))
        
        return warnings, direction, duration
    
    def get_duration(self, direction):
        """Get current duration for a specific direction"""
        current_time = time.time()
        if direction == "LEFT" and self.looking_left_start:
            return current_time - self.looking_left_start
        elif direction == "RIGHT" and self.looking_right_start:
            return current_time - self.looking_right_start
        elif direction == "DOWN" and self.looking_down_start:
            return current_time - self.looking_down_start
        return 0

# ====================== DETECTOR CLASSES ======================
class BlinkDetector:
    def __init__(self):
        self.blink_start = None
        self.is_blinking = False
        self.drowsy_counter = 0
        
    def check(self, ear, current_time, is_night):
        eye_thresh = NIGHT_EYE_THRESH if is_night else EYE_THRESH
        
        if ear < eye_thresh and not self.is_blinking:
            self.is_blinking = True
            self.blink_start = current_time
            return "OPEN", 0
        elif ear >= eye_thresh and self.is_blinking:
            self.is_blinking = False
            duration = current_time - self.blink_start
            if duration < BLINK_THRESH:
                return "NORMAL_BLINK", duration
            else:
                return "LONG_BLINK", duration
        elif self.is_blinking:
            duration = current_time - self.blink_start
            if duration > DANGEROUS_CLOSURE_THRESH:
                return "MICROSLEEP", duration
            return "BLINKING", duration
        
        if ear < eye_thresh:
            self.drowsy_counter += 1
        else:
            self.drowsy_counter = max(0, self.drowsy_counter - 2)
        
        return "OPEN", 0
    
    def is_drowsy(self):
        return self.drowsy_counter > DROWSY_CONSECUTIVE_FRAMES

class MouthDetector:
    def __init__(self):
        self.mar_history = deque(maxlen=15)
        self.yawn_start = None
        self.is_yawning = False
        self.yawn_counter = 0
        
    def check(self, mar, current_time):
        self.mar_history.append(mar)
        avg_mar = np.mean(self.mar_history)
        
        if avg_mar > MOUTH_THRESH:
            if not self.is_yawning:
                self.is_yawning = True
                self.yawn_start = current_time
                return "YAWN_START"
            else:
                duration = current_time - self.yawn_start
                if duration > YAWN_THRESH:
                    self.yawn_counter += 1
                    return "YAWNING"
                return "YAWNING"
        else:
            if self.is_yawning:
                self.is_yawning = False
            return "NORMAL"
    
    def is_excessive_yawn(self):
        return self.yawn_counter > 3

class HeadDetector:
    def estimate(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        nose = landmarks[1]
        left_eye = np.mean(landmarks[LEFT_EYE], axis=0)
        right_eye = np.mean(landmarks[RIGHT_EYE], axis=0)
        
        face_center_x = (left_eye[0] + right_eye[0]) / 2
        turn_amount = (nose[0] - face_center_x) / w
        
        chin = landmarks[152]
        forehead = landmarks[10]
        face_height = abs(forehead[1] - chin[1])
        nose_position = (nose[1] - forehead[1]) / face_height
        looking_down = nose_position > 0.7
        
        return turn_amount, looking_down

# ====================== MAIN MONITORING CLASS ======================
class DriverMonitor:
    def __init__(self):
        self.blink = BlinkDetector()
        self.mouth = MouthDetector()
        self.head = HeadDetector()
        self.timer = TimerBasedDetector()
        self.light = LightAnalyzer()
        self.sound = SoundAlert()
        
        self.active_warnings = set()
        self.warning_start_times = {}
        self.last_alarm_time = {}
        self.no_face_start_time = None
        self.risk_history = deque(maxlen=20)
        self.risk_score = 0
        self.primary_warning = "NORMAL"
        
    def play_alert_sound(self, warning_type, is_face_alert=False):
        current_time = time.time()
        
        if warning_type in self.last_alarm_time:
            if current_time - self.last_alarm_time[warning_type] < ALARM_COOLDOWN:
                return
        
        warning_info = WARNING_TYPES.get(warning_type, {})
        sound_type = warning_info.get('sound', 'beep')
        
        if sound_type == 'beep':
            self.sound.beep(1000, 400, is_face_alert)
        elif sound_type == 'double_beep':
            self.sound.double_beep()
        elif sound_type == 'continuous':
            self.sound.continuous_beep(3)
        
        self.last_alarm_time[warning_type] = current_time
    
    def process_frame(self, frame, landmarks, face_detected):
        current_time = time.time()
        warnings = []
        
        # Analyze lighting
        is_night, brightness = self.light.analyze(frame)
        
        # Face not detected
        if not face_detected:
            if self.no_face_start_time is None:
                self.no_face_start_time = current_time
            
            no_face_duration = current_time - self.no_face_start_time
            no_face_thresh = NIGHT_NO_FACE_TIME if is_night else NO_FACE_ALERT_TIME
            
            if no_face_duration > 1.5:  # Quick alert after 1.5 seconds
                if no_face_duration > no_face_thresh:
                    warnings.append(('NO_FACE', current_time))
            
            self.active_warnings = set([w[0] for w in warnings])
            self.calculate_risk_score()
            self.determine_primary_warning()
            
            # Play faster face alert sound
            if self.primary_warning == 'NO_FACE':
                self.sound.fast_face_alert()
            
            return {
                'face_detected': False,
                'is_night': is_night,
                'brightness': brightness,
                'warnings': list(self.active_warnings),
                'primary_warning': self.primary_warning,
                'risk_score': self.risk_score,
                'no_face_duration': no_face_duration
            }
        
        # Face detected - reset
        self.no_face_start_time = None
        
        # Get landmarks
        if landmarks is None or len(landmarks) == 0:
            return self.process_frame(frame, np.array([]), False)
        
        h, w = landmarks.shape[:2]
        
        # Calculate EAR
        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]
        
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Calculate MAR
        mouth = landmarks[MOUTH]
        mar = self.calculate_mar(mouth)
        
        # Get head pose
        turn_amount, looking_down = self.head.estimate(landmarks, (h, w))
        
        # Check states
        eye_state, eye_duration = self.blink.check(ear, current_time, is_night)
        mouth_state = self.mouth.check(mar, current_time)
        
        # Timer-based warnings (look left/right/down) - THIS IS THE KEY FEATURE
        timer_warnings, direction, duration = self.timer.update_head_direction(
            turn_amount, looking_down, current_time, is_night
        )
        warnings.extend(timer_warnings)
        
        # Microsleep
        if eye_state == "MICROSLEEP":
            warnings.append(('MICROSLEEP', current_time))
        
        # Drowsy
        elif self.blink.is_drowsy():
            warnings.append(('DROWSY', current_time))
        
        # Yawning
        if mouth_state == "YAWNING" and self.mouth.is_excessive_yawn():
            warnings.append(('YAWNING', current_time))
        
        # Update warnings
        self.update_warnings(warnings, current_time)
        self.calculate_risk_score()
        self.determine_primary_warning()
        
        # Play alert sound
        if self.primary_warning != "NORMAL":
            self.play_alert_sound(self.primary_warning)
        
        return {
            'face_detected': True,
            'ear': ear,
            'mar': mar,
            'eye_state': eye_state,
            'mouth_state': mouth_state,
            'direction': direction,
            'duration': duration,
            'turn_amount': turn_amount,
            'is_night': is_night,
            'brightness': brightness,
            'warnings': list(self.active_warnings),
            'primary_warning': self.primary_warning,
            'risk_score': self.risk_score,
            'no_face_duration': 0
        }
    
    def calculate_ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C + 0.001)
    
    def calculate_mar(self, mouth):
        vert1 = np.linalg.norm(mouth[2] - mouth[3])
        vert2 = np.linalg.norm(mouth[4] - mouth[5])
        horiz = np.linalg.norm(mouth[0] - mouth[1])
        return (vert1 + vert2) / (2.0 * horiz + 0.001)
    
    def update_warnings(self, new_warnings, current_time):
        for warning, time in new_warnings:
            self.active_warnings.add(warning)
            if warning not in self.warning_start_times:
                self.warning_start_times[warning] = time
        
        to_remove = []
        for warning in self.active_warnings:
            if warning not in [w[0] for w in new_warnings]:
                if current_time - self.warning_start_times.get(warning, 0) > 3.0:
                    to_remove.append(warning)
        
        for warning in to_remove:
            self.active_warnings.remove(warning)
    
    def calculate_risk_score(self):
        priority_sum = 0
        for warning in self.active_warnings:
            priority_sum += WARNING_TYPES.get(warning, {}).get('priority', 0)
        raw_risk = min(100, priority_sum * 8)
        self.risk_history.append(raw_risk)
        self.risk_score = int(np.mean(self.risk_history)) if self.risk_history else 0
    
    def determine_primary_warning(self):
        if not self.active_warnings:
            self.primary_warning = "NORMAL"
            return
        self.primary_warning = max(self.active_warnings, 
                                   key=lambda w: WARNING_TYPES.get(w, {}).get('priority', 0))

# ====================== DRAW UI FUNCTION ======================
def draw_ui(frame, data, points, fps):
    h, w = frame.shape[:2]
    
    # Draw landmarks if face detected
    if data['face_detected'] and points is not None:
        # Eyes
        eye_color = (0, 255, 0)
        if data['eye_state'] in ["MICROSLEEP", "DANGEROUS_CLOSURE"]:
            eye_color = (0, 0, 255)
        for point in points[LEFT_EYE].astype(int):
            cv2.circle(frame, tuple(point), 2, eye_color, -1)
        for point in points[RIGHT_EYE].astype(int):
            cv2.circle(frame, tuple(point), 2, eye_color, -1)
        
        # Mouth
        mouth_color = (0, 0, 255) if data['mouth_state'] == "YAWNING" else (100, 100, 100)
        for point in points[MOUTH].astype(int):
            cv2.circle(frame, tuple(point), 2, mouth_color, -1)
        
        # Head direction arrow with timer visualization
        nose = points[1].astype(int)
        direction = data['direction']
        
        # Color code based on duration
        if direction == "LEFT" and data['duration'] > 2.0:
            color = (0, 0, 255)  # Red if close to warning
        elif direction == "RIGHT" and data['duration'] > 2.0:
            color = (0, 0, 255)
        elif direction == "DOWN" and data['duration'] > 3.0:
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)  # Yellow for normal movement
        
        # Draw direction arrow
        if direction != "CENTER":
            end_point = (nose[0] + int(data['turn_amount'] * w * 2), nose[1])
            cv2.arrowedLine(frame, tuple(nose), end_point, color, 2)
            
            # Show timer if looking away (THIS SHOWS THE COUNTDOWN)
            if data['duration'] > 1.0:
                timer_text = f"{data['duration']:.1f}s"
                timer_color = (0, 0, 255) if data['duration'] > 2.5 else (255, 255, 0)
                cv2.putText(frame, timer_text, (nose[0] + 20, nose[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, timer_color, 2)
    
    # Top status bar
    cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 0), -1)
    
    # Light indicator
    light_color = (0, 255, 0) if not data['is_night'] else (0, 255, 255)
    cv2.putText(frame, f"Light: {data['brightness']:.0f}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, light_color, 1)
    
    if data['is_night']:
        cv2.putText(frame, "NIGHT MODE", (120, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Primary warning
    if data['primary_warning'] != "NORMAL":
        warning_info = WARNING_TYPES.get(data['primary_warning'], WARNING_TYPES['DROWSY'])
        cv2.putText(frame, warning_info['message'], (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, warning_info['color'], 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps}", (w - 80, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Left panel
    panel_x, panel_y = 10, 55
    panel_w, panel_h = 280, 180
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 1)
    
    cv2.putText(frame, "DRIVER STATUS", (panel_x + 10, panel_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    y_offset = 50
    if data['face_detected']:
        status_text = "ALERT" if data['primary_warning'] == "NORMAL" else "CAUTION"
        status_color = (0, 255, 0) if data['primary_warning'] == "NORMAL" else (0, 255, 255)
        cv2.putText(frame, f"Status: {status_text}", (panel_x + 15, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        y_offset += 25
        
        # Head direction with timer (SHOWS DURATION IN PANEL)
        if data['direction'] != "CENTER":
            dir_text = f"Looking: {data['direction']} ({data['duration']:.1f}s)"
            dir_color = (0, 0, 255) if data['duration'] > 2.5 else (255, 255, 0)
            cv2.putText(frame, dir_text, (panel_x + 15, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, dir_color, 1)
            y_offset += 20
        else:
            cv2.putText(frame, "Looking: STRAIGHT", (panel_x + 15, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y_offset += 20
        
        cv2.putText(frame, f"Eye: {data['ear']:.2f}", (panel_x + 15, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(frame, f"Mouth: {data['mar']:.2f}", (panel_x + 15, panel_y + y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "NO FACE DETECTED", (panel_x + 15, panel_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if data['no_face_duration'] > 1.0:
            cv2.putText(frame, f"Missing: {data['no_face_duration']:.1f}s", 
                       (panel_x + 15, panel_y + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    
    # Risk meter
    meter_x = panel_x + 20
    meter_y = panel_y + panel_h + 10
    meter_w = panel_w - 40
    meter_h = 20
    
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (50, 50, 50), -1)
    risk_width = int((data['risk_score'] / 100) * meter_w)
    risk_color = (0, 255, 0) if data['risk_score'] < 30 else (0, 255, 255) if data['risk_score'] < 60 else (0, 0, 255)
    if risk_width > 0:
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + risk_width, meter_y + meter_h), risk_color, -1)
    cv2.putText(frame, f"RISK: {data['risk_score']}%", (meter_x, meter_y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Active warnings
    if data['warnings']:
        y_offset = meter_y + meter_h + 20
        cv2.putText(frame, "ACTIVE WARNINGS:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y_offset += 20
        for warning in data['warnings'][:3]:
            info = WARNING_TYPES.get(warning, {})
            cv2.putText(frame, f"• {info.get('message', warning)}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, info.get('color', (255, 255, 255)), 1)
            y_offset += 18
    
    # Timer guide with night mode adjustment
    guide_y = h - 70
    cv2.putText(frame, "TIMER GUIDE:", (10, guide_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    if data['is_night']:
        cv2.putText(frame, "Left/Right: 3.5s | Down: 5s (Night)", (10, guide_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    else:
        cv2.putText(frame, "Left/Right: 3s | Down: 4s", (10, guide_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    cv2.putText(frame, "Q: Quit | A: Acknowledge", (w - 210, h - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Face not visible overlay
    if not data['face_detected'] and data['no_face_duration'] > 1.0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        msg = "FACE NOT VISIBLE"
        if int(time.time() * 4) % 2 == 0:
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            cv2.putText(frame, msg, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if data['no_face_duration'] > 1.0:
                cv2.putText(frame, f"{data['no_face_duration']:.0f}s", 
                           (text_x + text_size[0] + 20, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return frame

# ====================== MAIN FUNCTION ======================
def main():
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        input("Press Enter to exit...")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Camera ready")
    
    monitor = DriverMonitor()
    
    fps = 0
    fps_counter = 0
    fps_time = time.time()
    frame_count = 0
    
    print("\n" + "="*70)
    print("DRIVER MONITORING SYSTEM - TIMER-BASED WITH NIGHT MODE")
    print("="*70)
    print("\nTimer-Based Head Movement Detection:")
    print("  • Look left for 3+ seconds → Warning")
    print("  • Look right for 3+ seconds → Warning")
    print("  • Look down for 4+ seconds → Phone warning")
    print("  • Timer shows duration while looking away")
    print("\nNight Mode (automatic):")
    print("  • Extended timers: Left/Right 3.5s, Down 5s")
    print("  • More forgiving eye closure threshold")
    print("\nControls:")
    print("  Q - Quit | A - Acknowledge warnings")
    print("="*70)
    print("\nSystem running...\n")
    
    cv2.namedWindow("Driver Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Driver Monitor", 800, 600)
    
    last_face_detection_time = time.time()
    points = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error...", end="\r")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        frame = cv2.flip(frame, 1)
        
        # Process every Nth frame
        if frame_count % FRAME_SKIP == 0:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp = int(time.time() * 1000)
                
                detection = landmarker.detect_for_video(mp_image, timestamp)
                
                if detection.face_landmarks:
                    last_face_detection_time = time.time()
                    landmarks = detection.face_landmarks[0]
                    h, w, _ = frame.shape
                    points = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
                    face_detected = True
                else:
                    face_detected = False
                    points = None
                    
            except Exception as e:
                print(f"Detection error: {e}", end="\r")
                face_detected = False
                points = None
        else:
            face_detected = (points is not None and 
                            (time.time() - last_face_detection_time) < 0.5)
        
        # Process data
        data = monitor.process_frame(frame, points if points is not None else np.array([]), face_detected)
        
        # Draw UI
        display_frame = draw_ui(frame, data, points, fps)
        
        # Update FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        cv2.imshow("Driver Monitor", display_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('a') or key == ord('A'):
            monitor.active_warnings.clear()
            monitor.risk_history.clear()
            monitor.risk_score = 0
            print("\nWarnings acknowledged")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nSystem stopped")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")