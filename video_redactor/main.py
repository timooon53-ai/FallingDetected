# main.py
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from gui import VideoEditorGUI


class VideoEditor:
    def __init__(self):
        self.gui = VideoEditorGUI()
        self.cap = None  # Видеозахват
        self.fps = 0
        self.total_frames = 0
        self.current_frame = 0
        self.is_playing = False

        # Подключение сигналов
        self.gui.load_btn.clicked.connect(self.load_video)
        self.gui.play_btn.clicked.connect(self.toggle_play)
        self.gui.stop_btn.clicked.connect(self.stop_play)
        self.gui.speed_slider.valueChanged.connect(self.update_speed_label)
        self.gui.export_btn.clicked.connect(self.export_video)

        # Кнопки установки текущего кадра
        self.gui.set_start_btn.clicked.connect(lambda: self.gui.start_frame_slider.setValue(self.current_frame))
        self.gui.set_end_btn.clicked.connect(lambda: self.gui.end_frame_slider.setValue(self.current_frame))

        # Таймер для воспроизведения
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    def load_video(self):
        file_path, _ = self.gui.ask_open_filename()
        if not file_path:
            return

        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.gui.show_error("Не удалось открыть видеофайл!")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Обновляем GUI
        self.gui.file_label.setText(Path(file_path).name)

        # Слайдеры кадров
        self.gui.start_frame_slider.setMaximum(self.total_frames - 1)
        self.gui.end_frame_slider.setMaximum(self.total_frames - 1)
        self.gui.start_frame_slider.setValue(0)
        self.gui.end_frame_slider.setValue(self.total_frames - 1)

        self.current_frame = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.show_current_frame()

    def show_current_frame(self):
        if self.cap is None:
            return

        # Считываем конкретный кадр
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return

        # Конвертируем BGR → RGB для корректного отображения
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.gui.show_frame(frame_rgb)

    def next_frame(self):
        if self.cap is None or not self.is_playing:
            return
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.current_frame = 0  # Зацикливание
        self.show_current_frame()

    def toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.gui.play_btn.setText("▶ Воспроизвести")
            self.is_playing = False
        else:
            speed_factor = self.gui.get_speed_factor()
            delay = int(1000 / (self.fps * speed_factor))
            self.timer.start(delay)
            self.gui.play_btn.setText("II Пауза")
            self.is_playing = True

    def stop_play(self):
        self.timer.stop()
        self.gui.play_btn.setText("▶ Воспроизвести")
        self.is_playing = False
        self.current_frame = 0
        if self.cap:
            self.show_current_frame()

    def update_speed_label(self, value):
        speed_factor = value / 100.0
        self.gui.update_speed_label(speed_factor)

    def export_video(self):
        if self.cap is None:
            self.gui.show_error("Нет загруженного видео!")
            return

        # Получаем кадры из слайдеров
        start_frame = self.gui.start_frame_slider.value()
        end_frame = self.gui.end_frame_slider.value()
        speed_factor = self.gui.get_speed_factor()

        if start_frame >= end_frame:
            self.gui.show_error("Начальный кадр должен быть меньше конечного!")
            return

        # Запрашиваем путь сохранения
        output_path, _ = self.gui.ask_save_filename()
        if not output_path:
            return

        # Открываем видео для записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps * speed_factor,
            (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        try:
            for frame_idx in range(start_frame, end_frame):
                ret, frame = self.cap.read()
                if not ret:
                    break
                # Дублируем кадры при уменьшенной скорости
                for _ in range(max(1, int(1 / speed_factor))):
                    out.write(frame)
            out.release()
            QMessageBox.information(self.gui, "Успех", "Видео успешно экспортировано!")
        except Exception as e:
            self.gui.show_error(f"Ошибка при экспорте: {str(e)}")
        finally:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0

    def run(self):
        self.gui.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = VideoEditor()
    editor.run()
    sys.exit(app.exec())
