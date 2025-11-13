import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sqlite3
import matplotlib.pyplot as plt

video_path = r"C:\Users\Vitya\PycharmProjects\FallingBabka\бабка падает.mp4"
base_dir = os.path.dirname(video_path)

output_path = os.path.join(base_dir, "output_fall_status.mp4")
fall_dir = os.path.join(base_dir, "fall_screens")
os.makedirs(fall_dir, exist_ok=True)
db_path = os.path.join(base_dir, "fall_events.db")

# === Подключаем базу данных ===
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS fall_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fall_frame INTEGER,
    up_frame INTEGER,
    duration_frames INTEGER,
    fall_image TEXT,
    up_image TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# === MediaPipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Не удалось открыть видео:", video_path)
    exit()

# === Настройки видео ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === Переменные анализа ===
prev_y = None
prev_time = None
fall_detected = False
fall_frame = None
fall_img_saved = False
lying_frames = 0
frame_count = 0

# === Пороговые значения ===
speed_threshold = 0.15
amplitude_threshold = 0.18
angle_threshold = 0.06
lying_min_frames = 15

# === Для графиков ===
frames_list = []
speed_list = []
angle_list = []
height_list = []

print("▶ Начало обработки видео с фиксацией статуса...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    current_time = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    status_text = "СТОИТ"
    status_color = (0, 255, 0)  # Зеленый по умолчанию

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        body_angle = abs(shoulder_y - hip_y)

        dy, speed = 0, 0
        if prev_y is not None and prev_time is not None:
            dy = shoulder_y - prev_y
            dt = current_time - prev_time
            speed = dy / dt if dt > 0 else 0

        prev_y = shoulder_y
        prev_time = current_time

        # === Данные для графиков ===
        frames_list.append(frame_count)
        speed_list.append(speed)
        angle_list.append(body_angle)
        height_list.append(shoulder_y)

        # === Определение падения ===
        if not fall_detected and speed > speed_threshold and dy > amplitude_threshold and body_angle < angle_threshold:
            fall_detected = True
            fall_frame = frame_count
            lying_frames = 0
            fall_img_saved = False

        # === Определяем состояние ===
        if fall_detected:
            status_text = "БАБКА УПАЛА!"
            status_color = (0, 0, 255)  # Красный
            lying_frames += 1

            # === Сохраняем скриншот один раз ===
            if not fall_img_saved:
                fall_img_path = os.path.join(fall_dir, f"falling_{fall_frame}.jpg")
                img_copy = frame.copy()
                cv2.putText(img_copy, "БАБКА УПАЛА!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
                cv2.imwrite(fall_img_path, img_copy)
                fall_img_saved = True

                # Сохраняем факт падения в БД сразу
                cur.execute("""
                    INSERT INTO fall_events (fall_frame, up_frame, duration_frames, fall_image, up_image)
                    VALUES (?, ?, ?, ?, ?)
                """, (fall_frame, None, None, fall_img_path, None))
                conn.commit()

        elif body_angle < angle_threshold:  # лежит/сидит
            status_text = "СИДИТ/ЛЕЖИТ"
            status_color = (0, 255, 255)  # Желтый
            lying_frames += 1
        else:
            status_text = "СТОИТ"
            status_color = (0, 255, 0)
            lying_frames = 0
            fall_detected = False

        # === Отрисовка текста на видео ===
        cv2.putText(frame, f"{status_text}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        cv2.putText(frame, f"Speed: {speed:.3f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Angle: {body_angle:.3f}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"dy: {dy:.3f}", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # === Отрисовка скелета ===
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=status_color, thickness=3, circle_radius=4),
            mp_drawing.DrawingSpec(color=status_color, thickness=2)
        )

    out.write(frame)
    cv2.imshow("Fall Detection Status", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
conn.close()
cv2.destroyAllWindows()

print("\n✅ Обработка завершена!")
print(f"🎥 Видео: {output_path}")
print(f"💾 База данных: {db_path}")
print(f"📸 Скриншоты: {fall_dir}")
