import cv2
import os
import time
import sqlite3
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

video_path = r"C:\Users\Vitya\PycharmProjects\FallingBabka\скейт падение .mp4" #видео путь
base_dir = os.path.dirname(video_path)

"""
зеленый цвет - бабка в норме, с ней ничего не происходит
желтый цвет - бабка лежит, но пока не умерла
красный цвет - бабка лежит неподвижно, поэтому, по хорошему бы ее поднять
"""
#конфигурация файлов
output_path = os.path.join(base_dir, "yolo.mp4")
fall_dir = os.path.join(base_dir, "fall_screens")
os.makedirs(fall_dir, exist_ok=True)
db_path = os.path.join(base_dir, "fall_events.db")


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

#загрузка модели
model = YOLO("yolov8n-pose.pt")

#model2 = YOLO("yolov8n-seg.pt")

#настройка видео вывода
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Ошибка откртыия видео:", video_path)
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

prev_y = None
prev_time = None
fall_detected = False
fall_frame = None
frame_count = 0
lying_frames = 0
is_lying = False

#параметры
speed_threshold = 0.15
amplitude_threshold = 0.18
angle_threshold = 0.06
lying_min_frames = 5

#график
frames_list = []
speed_list = []
angle_list = []
height_list = []

print("ЗАПУЩЕНО")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    current_time = time.time()

    results = model.predict(frame, verbose=False)
    #results2 = model2.predict(frame, verbose=False)
    if len(results) > 0 and len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()

        if keypoints.shape[0] > 12:
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]

            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2 / height
            hip_y = (left_hip[1] + right_hip[1]) / 2 / height
            body_angle = abs(shoulder_y - hip_y)

            #анализатор
            dy, speed = 0, 0
            if prev_y is not None and prev_time is not None:
                dy = shoulder_y - prev_y
                dt = current_time - prev_time
                speed = dy / dt if dt > 0 else 0

            prev_y = shoulder_y
            prev_time = current_time

            #добавление в график падения безумной бабки
            frames_list.append(frame_count)
            speed_list.append(speed)
            angle_list.append(body_angle)
            height_list.append(shoulder_y)

            text_state = "Стоит"
            color = (0, 255, 0)

            #падение
            if not fall_detected and speed > speed_threshold and dy > amplitude_threshold and body_angle < angle_threshold:
                fall_detected = True
                fall_frame = frame_count
                lying_frames = 0
                fall_img_path = os.path.join(fall_dir, f"falling_{fall_frame}.jpg")
                cv2.imwrite(fall_img_path, frame)
                print(f"Бабка упала на кадре {fall_frame}")
                text_state = "ПАДЕНИЕ!"
                color = (0, 0, 255)

            #бабка лежит
            if body_angle < angle_threshold:
                lying_frames += 1
                if lying_frames >= lying_min_frames and not is_lying:
                    is_lying = True
                    print(f"Бабка лежит с кадра {frame_count}")

                    #Сохранение события падения
                    cur.execute("""
                        INSERT INTO fall_events (fall_frame, up_frame, duration_frames, fall_image, up_image)
                        VALUES (?, ?, ?, ?, ?)
                    """, (fall_frame or frame_count, None, lying_frames,
                          fall_img_path if fall_detected else None, None))
                    conn.commit()
            else:
                lying_frames = 0

            #текст на видео
            cv2.putText(frame, f"Speed: {speed:.3f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Angle: {body_angle:.3f}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"dy: {dy:.3f}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if fall_detected:
                text_state = "ЧЕЛОВЕК УПАЛ!"
                color = (0, 0, 255)
            elif is_lying:
                text_state = "ЧЕЛОВЕК ЛЕЖИТ"
                color = (0, 255, 255)

            cv2.putText(frame, text_state, (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Рисуем скелет
            for (x, y) in keypoints:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    out.write(frame)
    cv2.imshow("YOLOv8 Fall Detection v3", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
conn.close()
cv2.destroyAllWindows()

print(f"Сохранено видео: {output_path}")
print(f"База данных: {db_path}")
print(f"Скриншоты: {fall_dir}")

#график построение
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(frames_list, height_list)
plt.title("Положение плеча (высота)")
plt.ylabel("Нормализованная высота")

plt.subplot(3, 1, 2)
plt.plot(frames_list, speed_list)
plt.title("Скорость движения")
plt.ylabel("Speed")

plt.subplot(3, 1, 3)
plt.plot(frames_list, angle_list)
plt.title("Угол тела (вертикальность)")
plt.ylabel("Angle")
plt.xlabel("Кадр")

plt.tight_layout()
plt.show()
