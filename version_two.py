import cv2
import mediapipe as mp
import numpy as np
import os
import sqlite3
from datetime import timedelta

VIDEO_PATH = r"C:\Users\Vitya\PycharmProjects\FallingBabka\3\скейт падение .mp4"
OUT_VIDEO_PATH = r"C:\Users\Vitya\PycharmProjects\FallingBabka\6.mp4"

OUTPUT_FRAMES_DIR = "frames_positions6"
DB_PATH = "fall_events6.db"
LOG_PATH = "fall_events_log6.txt"

HORIZONTAL_ANGLE_THRESHOLD = 40  #угол < 40° считается лежащим
VERTICAL_ANGLE_THRESHOLD = 50    #угол > 90-50 = 40° считается стоящим
HOLD_FRAMES = 1                   #кадров подряд для фиксации падения
UP_HOLD_FRAMES = 10               #кадров подряд для фиксации подъёма
MIN_VISIBILITY = 0.3

os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
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

init_db()

mp_pose = mp.solutions.pose

def torso_angle_deg(landmarks, w, h):
    """Угол торса между средними плечами и бёдрами"""
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

def save_event(fall_time_text, fall_time_sec, fall_frame, stand_time_text, stand_time_sec, stand_frame, duration):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events (fall_time, fall_time_sec, fall_frame, stand_time, stand_time_sec, stand_frame, duration) VALUES (?,?,?,?,?,?,?)",
        (fall_time_text, fall_time_sec, fall_frame, stand_time_text, stand_time_sec, stand_frame, duration)
    )
    conn.commit()
    conn.close()

    # текстовый лог
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"FALL: {fall_time_text} sec={fall_time_sec}\n")
            f.write(f"  frame: {fall_frame}\n")
            f.write(f"STAND: {stand_time_text} sec={stand_time_sec}\n")
            f.write(f"  frame: {stand_frame}\n")
            f.write(f"DURATION lying: {duration} sec\n")
            f.write("-------------------------------------\n")
    except Exception as e:
        print("Ошибка записи лога:", e)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, (frame_w, frame_h))

frame_count = 0
fall_counter = 0
stand_counter = 0
fallen = False
fall_time_sec = None
fall_frame_path = None

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

with mp_pose.Pose(min_detection_confidence=0.65, min_tracking_confidence=0.5) as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        sec = frame_count / fps
        timecode = str(timedelta(seconds=sec))
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        state_label = "NoPerson"
        angle_deg = 0
        color = (255, 255, 255)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            if avg_visibility(lm) < MIN_VISIBILITY:
                state_label = "LowVisibility"
                color = (255, 255, 255)
            else:

                angle_deg, shoulder_pt, hip_pt = torso_angle_deg(lm, w, h)
                abs_angle = abs(angle_deg)
                horiz_angle = min(abs_angle, abs(180 - abs_angle))

                is_horizontal = horiz_angle < HORIZONTAL_ANGLE_THRESHOLD
                is_vertical = horiz_angle > (90 - VERTICAL_ANGLE_THRESHOLD)


                if is_horizontal:
                    fall_counter += 1
                    stand_counter = 0
                elif is_vertical:
                    stand_counter += 1
                    fall_counter = 0
                else:
                    fall_counter = 0
                    stand_counter = 0

                if fall_counter >= HOLD_FRAMES and not fallen:
                    fallen = True
                    fall_time_sec = sec
                    fall_timecode_text = timecode
                    fall_frame_path = os.path.join(OUTPUT_FRAMES_DIR, f"fall_{frame_count}.jpg")
                    cv2.imwrite(fall_frame_path, frame)
                    print(f"Падение: {fall_timecode_text}")


                if fallen and stand_counter >= UP_HOLD_FRAMES:
                    stand_time_sec = sec
                    stand_timecode_text = timecode
                    stand_frame_path = os.path.join(OUTPUT_FRAMES_DIR, f"stand_{frame_count}.jpg")
                    cv2.imwrite(stand_frame_path, frame)

                    duration = stand_time_sec - fall_time_sec
                    save_event(fall_timecode_text, fall_time_sec, fall_frame_path,
                               stand_timecode_text, stand_time_sec, stand_frame_path, duration)
                    print(f"Подъём: {stand_timecode_text}, duration={duration}")

                    fallen = False
                    fall_counter = 0
                    stand_counter = 0
                    fall_time_sec = None
                    fall_frame_path = None


                if fallen:
                    color = (0, 0, 255)  # красный
                    state_label = f"Fallen ({horiz_angle:.1f}°)"
                elif is_horizontal:
                    color = (0, 255, 255)  # жёлтый
                    state_label = f"Lying ({horiz_angle:.1f}°)"
                elif is_vertical:
                    color = (0, 255, 0)  # зелёный
                    state_label = f"Standing ({horiz_angle:.1f}°)"
                else:
                    color = (255, 255, 255)  # белый
                    state_label = f"Unknown ({horiz_angle:.1f}°)"

                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                                 circle_radius=3),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                                                   circle_radius=2)
                )

                cv2.line(frame,
                         (int(shoulder_pt[0]), int(shoulder_pt[1])),
                         (int(hip_pt[0]), int(hip_pt[1])),
                         color, 3)

                arrow_len = 50
                dx = hip_pt[0] - shoulder_pt[0]
                dy = hip_pt[1] - shoulder_pt[1]
                norm = np.sqrt(dx * dx + dy * dy)
                if norm > 1e-3:
                    dx_norm = dx / norm * arrow_len
                    dy_norm = dy / norm * arrow_len
                    cv2.arrowedLine(frame,
                                    (int(shoulder_pt[0]), int(shoulder_pt[1])),
                                    (int(shoulder_pt[0] + dx_norm), int(shoulder_pt[1] + dy_norm)),
                                    color, 2, tipLength=0.3)

        cv2.putText(frame, f"{state_label} Angle={angle_deg:.1f}°", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"{timecode} f={frame_count}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(OUTPUT_FRAMES_DIR, f"pos_{frame_count}.jpg"), frame)
        out_writer.write(frame)
        cv2.imshow("Processing", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

out_writer.release()
cap.release()
cv2.destroyAllWindows()
print("Готово. Видео и события сохранены.")
