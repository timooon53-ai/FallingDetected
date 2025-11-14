import cv2
import mediapipe as mp
import numpy as np
import os
import sqlite3
from datetime import timedelta
from ultralytics import YOLO

VIDEO_PATH = r"C:\Users\Vitya\PycharmProjects\FallingBabka\скейт падение .mp4"


"""YOLO НЕ РАБОТАЕТ"""

MP_VIDEO = r"C:\Users\Vitya\PycharmProjects\FallingBabka\MP_out.mp4"
MP_FRAMES = "MP_frames"
MP_DB = "MP_fall_events.db"
MP_LOG = "MP_fall_events_log.txt"

YOLO_VIDEO = r"C:\Users\Vitya\PycharmProjects\FallingBabka\YOLO_out.mp4"
YOLO_FRAMES = "YOLO_frames"
YOLO_DB = "YOLO_fall_events.db"
YOLO_LOG = "YOLO_fall_events_log.txt"

os.makedirs(MP_FRAMES, exist_ok=True)
os.makedirs(YOLO_FRAMES, exist_ok=True)

HORIZONTAL_ANGLE_THRESHOLD = 40
VERTICAL_ANGLE_THRESHOLD = 50
HOLD_FRAMES = 1
UP_HOLD_FRAMES = 10
MIN_VISIBILITY = 0.3

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fall_time TEXT,
        fall_time_sec REAL,
        fall_frame TEXT,
        stand_time TEXT,
        stand_time_sec REAL,
        stand_frame TEXT,
        duration REAL
    )''')
    conn.commit()
    conn.close()

init_db(MP_DB)
init_db(YOLO_DB)

def save_event(fall_time_text, fall_time_sec, fall_frame,
               stand_time_text, stand_time_sec, stand_frame, duration,
               db_path, log_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events (fall_time, fall_time_sec, fall_frame, stand_time, stand_time_sec, stand_frame, duration) VALUES (?,?,?,?,?,?,?)",
        (fall_time_text, fall_time_sec, fall_frame, stand_time_text, stand_time_sec, stand_frame, duration)
    )
    conn.commit()
    conn.close()
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"FALL: {fall_time_text} sec={fall_time_sec}\n")
            f.write(f"  frame: {fall_frame}\n")
            f.write(f"STAND: {stand_time_text} sec={stand_time_sec}\n")
            f.write(f"  frame: {stand_frame}\n")
            f.write(f"DURATION lying: {duration} sec\n")
            f.write("-------------------------------------\n")
    except Exception as e:
        print("Ошибка записи лога:", e)

mp_pose = mp.solutions.pose

def torso_angle_deg(landmarks, w, h):
    L_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    R_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    L_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    R_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    sh_x = ((L_sh.x + R_sh.x)/2)*w
    sh_y = ((L_sh.y + R_sh.y)/2)*h
    hip_x = ((L_hip.x + R_hip.x)/2)*w
    hip_y = ((L_hip.y + R_hip.y)/2)*h

    dx = hip_x - sh_x
    dy = hip_y - sh_y
    angle = np.degrees(np.arctan2(dy, dx))
    return angle, (sh_x, sh_y), (hip_x, hip_y)

def avg_visibility(landmarks):
    keys = [mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP]
    vis = [landmarks[k].visibility for k in keys]
    return float(np.mean(vis))

yolo_model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
mp_writer = cv2.VideoWriter(MP_VIDEO, fourcc, fps, (frame_w, frame_h))
yolo_writer = cv2.VideoWriter(YOLO_VIDEO, fourcc, fps, (frame_w, frame_h))

frame_count = 0

mp_fall_counter = 0
mp_stand_counter = 0
mp_fallen = False
mp_fall_time_sec = None
mp_fall_frame_path = None

yolo_fall_counter = 0
yolo_stand_counter = 0
yolo_fallen = False
yolo_fall_time_sec = None
yolo_fall_frame_path = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as mp_pose_model:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        sec = frame_count / fps
        timecode = str(timedelta(seconds=sec))
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_frame = frame.copy()
        mp_state_label = "NoPerson"
        angle_deg = 0
        color = (255,255,255)

        mp_result = mp_pose_model.process(rgb)
        if mp_result.pose_landmarks:
            lm = mp_result.pose_landmarks.landmark
            if avg_visibility(lm) >= MIN_VISIBILITY:
                angle_deg, _, _ = torso_angle_deg(lm, w, h)
                abs_angle = abs(angle_deg)
                horiz_angle = min(abs_angle, 180-abs_angle)
                is_horizontal = horiz_angle < HORIZONTAL_ANGLE_THRESHOLD
                is_vertical = horiz_angle > (90-VERTICAL_ANGLE_THRESHOLD)

                if is_horizontal:
                    mp_fall_counter += 1
                    mp_stand_counter = 0
                elif is_vertical:
                    mp_stand_counter += 1
                    mp_fall_counter = 0
                else:
                    mp_fall_counter = 0
                    mp_stand_counter = 0

                if mp_fall_counter >= HOLD_FRAMES and not mp_fallen:
                    mp_fallen = True
                    mp_fall_time_sec = sec
                    mp_fall_timecode = timecode
                    mp_fall_frame_path = os.path.join(MP_FRAMES, f"fall_{frame_count}.jpg")
                    cv2.imwrite(mp_fall_frame_path, mp_frame)

                if mp_fallen and mp_stand_counter >= UP_HOLD_FRAMES:
                    stand_time_sec = sec
                    stand_time_text = timecode
                    stand_frame_path = os.path.join(MP_FRAMES, f"stand_{frame_count}.jpg")
                    cv2.imwrite(stand_frame_path, mp_frame)
                    duration = stand_time_sec - mp_fall_time_sec
                    save_event(mp_fall_timecode, mp_fall_time_sec, mp_fall_frame_path,
                               stand_time_text, stand_time_sec, stand_frame_path, duration,
                               MP_DB, MP_LOG)
                    mp_fallen = False
                    mp_fall_counter = 0
                    mp_stand_counter = 0
                    mp_fall_time_sec = None
                    mp_fall_frame_path = None

                if mp_fallen:
                    color = (0,0,255)
                    mp_state_label = f"Fallen ({horiz_angle:.1f}°)"
                elif is_horizontal:
                    color = (0,255,255)
                    mp_state_label = f"Lying ({horiz_angle:.1f}°)"
                elif is_vertical:
                    color = (0,255,0)
                    mp_state_label = f"Standing ({horiz_angle:.1f}°)"
                else:
                    color = (255,255,255)
                    mp_state_label = f"Unknown ({horiz_angle:.1f}°)"

                mp.solutions.drawing_utils.draw_landmarks(
                    mp_frame, mp_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                )

        cv2.putText(mp_frame, f"{mp_state_label} Angle={angle_deg:.1f}°", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        mp_writer.write(mp_frame)
        cv2.imwrite(os.path.join(MP_FRAMES,f"frame_{frame_count}.jpg"), mp_frame)

        yolo_frame = frame.copy()
        yolo_state_label = "NoPerson"
        angle_deg_yolo = 0
        color = (255,255,255)

        results = yolo_model.predict(yolo_frame, imgsz=640, task='pose', verbose=False)
        if results and len(results) > 0:
            r = results[0]
            if r.keypoints is not None and len(r.keypoints) > 0:
                kpts = r.keypoints[0].numpy()
                if kpts.shape[0] >= 13:
                    px_sh = (kpts[5,0]+kpts[6,0])/2
                    py_sh = (kpts[5,1]+kpts[6,1])/2
                    px_hip = (kpts[11,0]+kpts[12,0])/2
                    py_hip = (kpts[11,1]+kpts[12,1])/2
                    dx = px_hip - px_sh
                    dy = py_hip - py_sh
                    angle_deg_yolo = np.degrees(np.arctan2(dy, dx))
                    abs_angle = abs(angle_deg_yolo)
                    horiz_angle = min(abs_angle, 180-abs_angle)
                    is_horizontal = horiz_angle < HORIZONTAL_ANGLE_THRESHOLD
                    is_vertical = horiz_angle > (90-VERTICAL_ANGLE_THRESHOLD)

                    if is_horizontal:
                        yolo_fall_counter += 1
                        yolo_stand_counter = 0
                    elif is_vertical:
                        yolo_stand_counter += 1
                        yolo_fall_counter = 0
                    else:
                        yolo_fall_counter = 0
                        yolo_stand_counter = 0

                    if yolo_fall_counter >= HOLD_FRAMES and not yolo_fallen:
                        yolo_fallen = True
                        yolo_fall_time_sec = sec
                        yolo_fall_timecode = timecode
                        yolo_fall_frame_path = os.path.join(YOLO_FRAMES,f"fall_{frame_count}.jpg")
                        cv2.imwrite(yolo_fall_frame_path, yolo_frame)

                    if yolo_fallen and yolo_stand_counter >= UP_HOLD_FRAMES:
                        stand_time_sec = sec
                        stand_time_text = timecode
                        stand_frame_path = os.path.join(YOLO_FRAMES,f"stand_{frame_count}.jpg")
                        cv2.imwrite(stand_frame_path, yolo_frame)
                        duration = stand_time_sec - yolo_fall_time_sec
                        save_event(yolo_fall_timecode, yolo_fall_time_sec, yolo_fall_frame_path,
                                   stand_time_text, stand_time_sec, stand_frame_path, duration,
                                   YOLO_DB, YOLO_LOG)
                        yolo_fallen = False
                        yolo_fall_counter = 0
                        yolo_stand_counter = 0
                        yolo_fall_time_sec = None
                        yolo_fall_frame_path = None

                    if yolo_fallen:
                        color = (0,0,255)
                        yolo_state_label = f"Fallen ({horiz_angle:.1f}°)"
                    elif is_horizontal:
                        color = (0,255,255)
                        yolo_state_label = f"Lying ({horiz_angle:.1f}°)"
                    elif is_vertical:
                        color = (0,255,0)
                        yolo_state_label = f"Standing ({horiz_angle:.1f}°)"
                    else:
                        color = (255,255,255)
                        yolo_state_label = f"Unknown ({horiz_angle:.1f}°)"

                    yolo_frame = r.plot()

        cv2.putText(yolo_frame, f"{yolo_state_label} Angle={angle_deg_yolo:.1f}°", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color,2)
        yolo_writer.write(yolo_frame)
        cv2.imwrite(os.path.join(YOLO_FRAMES,f"frame_{frame_count}.jpg"), yolo_frame)

        combined = np.hstack([
            cv2.resize(mp_frame,(320,240)),
            cv2.resize(yolo_frame,(320,240))
        ])
        cv2.imshow("MediaPipe vs YOLO", combined)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
mp_writer.release()
yolo_writer.release()
cv2.destroyAllWindows()
print("Готово! Два видео, две БД, два лога и кадры сохранены.")
