from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QGroupBox, QFileDialog, QMessageBox, QFormLayout
)


class VideoEditorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Простой видеоредактор")
        self.resize(800, 600)
        self.video_path = None
        self.total_frames = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Верхняя панель: загрузка и информация
        top_group = QGroupBox("Видео")
        top_layout = QHBoxLayout()
        self.load_btn = QPushButton("Загрузить видео")
        top_layout.addWidget(self.load_btn)
        self.file_label = QLabel("Нет файла")
        top_layout.addWidget(self.file_label)
        top_group.setLayout(top_layout)
        layout.addWidget(top_group)

        # Превью видео
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.preview_label, 1)

        # Управление воспроизведением
        ctrl_group = QGroupBox("Управление")
        ctrl_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Воспроизвести")
        ctrl_layout.addWidget(self.play_btn)
        self.stop_btn = QPushButton("■ Стоп")
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        # Настройки обрезки и скорости
        settings_group = QGroupBox("Настройки редактирования")
        settings_layout = QFormLayout()

        # Слайдеры для выбора начала и конца
        self.start_frame_slider = QSlider(Qt.Horizontal)
        self.start_frame_slider.setMinimum(0)
        self.start_frame_slider.setMaximum(0)
        self.start_frame_slider.valueChanged.connect(self.update_start_label)
        self.start_label = QLabel("Начало: 0")

        self.end_frame_slider = QSlider(Qt.Horizontal)
        self.end_frame_slider.setMinimum(0)
        self.end_frame_slider.setMaximum(0)
        self.end_frame_slider.valueChanged.connect(self.update_end_label)
        self.end_label = QLabel("Конец: 0")

        settings_layout.addRow(self.start_label, self.start_frame_slider)
        settings_layout.addRow(self.end_label, self.end_frame_slider)

        # Кнопки для установки текущего кадра
        self.set_start_btn = QPushButton("Установить текущий кадр как начало")
        self.set_end_btn = QPushButton("Установить текущий кадр как конец")
        settings_layout.addRow(self.set_start_btn)
        settings_layout.addRow(self.set_end_btn)

        # Скорость
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 300)  # 0.1x — 3.0x
        self.speed_slider.setValue(100)  # 1.0x
        self.speed_label = QLabel("Скорость: 1.0x")
        settings_layout.addRow("Скорость:", self.speed_slider)
        settings_layout.addRow("", self.speed_label)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Экспорт
        export_group = QGroupBox("Экспорт")
        export_layout = QHBoxLayout()
        self.export_btn = QPushButton("Экспортировать видео")
        export_layout.addWidget(self.export_btn)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # Отображение кадра
    def show_frame(self, frame):
        if frame is None:
            return
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(pixmap.scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    # Метки для слайдеров
    def update_start_label(self, value):
        self.start_label.setText(f"Начало: {value}")

    def update_end_label(self, value):
        self.end_label.setText(f"Конец: {value}")

    # Методы для main.py
    def set_file_info(self, filename):
        """Устанавливает имя загруженного файла"""
        self.file_label.setText(filename)

    def get_speed_factor(self):
        """Возвращает коэффициент скорости воспроизведения"""
        return self.speed_slider.value() / 100.0

    def update_speed_label(self, factor):
        """Обновляет метку скорости"""
        self.speed_label.setText(f"Скорость: {factor:.1f}x")

    def show_error(self, message):
        QMessageBox.warning(self, "Ошибка", message)

    def ask_save_filename(self):
        return QFileDialog.getSaveFileName(self, "Сохранить видео", "", "Видео (*.mp4)")

    def ask_open_filename(self):
        return QFileDialog.getOpenFileName(self, "Выбрать видеофайл", "", "Видео (*.mp4 *.avi *.mov)")
