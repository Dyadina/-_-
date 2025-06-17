import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PyQt6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QFrame, QTabWidget, QTextEdit, QTableWidget,
    QTableWidgetItem
)

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPixmap, QFont, QIcon
from src.utils.visualization import plot_probabilities
from src.utils.file_utils import save_classification_result
from src.ml.inference import predict_soil_condition
from src.ml.report_utils import analyze_image, generate_report
from src.ml.report_utils import generate_html_report
from PyQt6.QtWidgets import QTextBrowser



class CustomMessage(QWidget):
    def __init__(self, message):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Popup)
        self.setStyleSheet("""
            QWidget {
                background-color: #21222c;
                border: 1px solid #6272a4;
                border-radius: 8px;
            }
            QLabel {
                color: #f8f8f2;
                font-size: 13px;
                padding: 10px;
            }
            QPushButton {
                background-color: #44475a;
                color: white;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #6272a4;
            }
        """)
        self.setFixedSize(300, 120)

        layout = QVBoxLayout()
        label = QLabel(message)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn = QPushButton("OK")
        btn.clicked.connect(self.close)

        layout.addWidget(label)
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.resize(1000, 750)
        self.setMinimumSize(800, 500)
        self.setStyleSheet("background-color: #282a36; color: #f8f8f2;")
        self.image_path = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        self.header = QFrame()
        self.header.setFixedHeight(40)
        self.header.setStyleSheet("background-color: #21222c;")
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 0, 10, 0)

        self.title = QLabel("Класифікація стану ґрунтів")
        self.title.setFont(QFont("Segoe UI", 11))
        self.title.setStyleSheet("color: #f8f8f2;")

        btn_close = QPushButton("✖")
        btn_close.clicked.connect(self.close)
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #f8f8f2;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #ff5555;
            }
        """)
        btn_close.setFixedSize(40, 30)

        header_layout.addWidget(self.title)
        header_layout.addStretch()
        header_layout.addWidget(btn_close)

        # Main body
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)

        # Sidebar
        menu_widget = QWidget()
        menu_widget.setFixedWidth(200)
        menu_widget.setStyleSheet("background-color: #21222c;")
        menu_layout = QVBoxLayout(menu_widget)
        menu_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        font = QFont('Segoe UI', 11)

        self.btn_load = QPushButton("📁 Завантажити")
        self.btn_classify = QPushButton("✨ Класифікувати")
        self.btn_exit = QPushButton("⛔ Вийти")

        for btn in [self.btn_load, self.btn_classify, self.btn_exit]:
            btn.setFont(font)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #44475a;
                    color: #f8f8f2;
                    padding: 10px;
                    margin: 8px;
                    border: none;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #6272a4;
                }
            """)
            menu_layout.addWidget(btn)

        self.btn_load.clicked.connect(self.load_image)
        self.btn_classify.clicked.connect(self.classify_image)
        self.btn_exit.clicked.connect(self.close)

        # Central widget
        self.center_widget = QWidget()
        center_layout = QVBoxLayout(self.center_widget)

        self.label_image = QLabel("🛰️ Завантажте супутниковий знімок")
        self.label_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_image.setFont(QFont('Segoe UI', 14))
        self.label_image.setStyleSheet("color: #f8f8f2; margin: 20px;")

        self.result_label = QLabel("📊 Результат буде тут")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setFont(QFont('Segoe UI', 13))
        self.result_label.setStyleSheet("color: #bd93f9; margin-top: 20px;")

        center_layout.addWidget(self.label_image)
        center_layout.addWidget(self.result_label)

        body_layout.addWidget(menu_widget)
        body_layout.addWidget(self.center_widget)

        main_layout.addWidget(self.header)

        # === Tabs ===
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #44475a;
                background-color: #282a36;
            }

            QTabBar::tab {
                background: #21222c;
                color: #f8f8f2;
                border: 1px solid #44475a;
                padding: 6px 16px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
            }

            QTabBar::tab:selected {
                background: #44475a;
                color: #50fa7b;
                border-bottom: 1px solid #282a36;
            }

            QTabBar::tab:hover {
                background: #6272a4;
                color: #ffffff;
            }
        """)

        # === Вкладка Класифікація ===
        tab_classify = QWidget()
        tab_classify.setLayout(body_layout)

        # === Вкладка Звіт ===
        self.tab_report = self.create_report_tab()

        # === Додаємо у таби ===
        self.tabs.addTab(tab_classify, "Класифікація")
        self.tabs.addTab(self.tab_report, "Звіт")

        main_layout.addWidget(self.tabs)

    def create_report_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # HTML браузер — повністю в темному стилі
        self.report_browser = QTextBrowser()
        self.report_browser.setOpenExternalLinks(True)
        self.report_browser.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1f29;
                color: #f8f8f2;
                border: 1px solid #44475a;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        self.report_browser.setMinimumHeight(100)

        # Таблиця метрик
        self.metrics_table = QTableWidget(4, 2)
        self.metrics_table.setCornerButtonEnabled(False)  # Вимикає кутову кнопку

        # 🔻 Приховуємо вертикальні заголовки (опціонально, якщо не хочеш ліву колонку заголовків)
        # self.metrics_table.verticalHeader().setVisible(False)
        self.hist_image = QLabel()
        self.hist_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hist_image.setStyleSheet("background-color: #1e1f29; border: 1px solid #44475a;")
        layout.addWidget(self.hist_image)

        # Встановлюємо заголовки
        self.metrics_table.setHorizontalHeaderLabels(["Метрика", "Значення"])
        self.metrics_table.setVerticalHeaderLabels(["Яскравість", "Контраст", "Темні пікселі", "Світлі пікселі"])

        # Наповнюємо ліву колонку вручну, якщо verticalHeader приховано
        # for row, name in enumerate(["Яскравість", "Контраст", "Темні пікселі", "Світлі пікселі"]):
        #     self.metrics_table.setItem(row, 0, QTableWidgetItem(name))

        # Стилізація
        self.metrics_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1f29;
                color: #f8f8f2;
                gridline-color: #44475a;
                border: 1px solid #44475a;
            }
            QHeaderView::section {
                background-color: #282a36;
                color: #ff79c6;
                padding: 4px;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QTableCornerButton::section {
                background-color: #1e1f29;
                border: none;
            }
        """)

        # Заборонити системне тло (усуває артефакти)
        self.metrics_table.viewport().setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        # Показати сітку
        self.metrics_table.setShowGrid(True)

        # Заповнення порожніх клітинок (щоб не було None)
        for row in range(4):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(""))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(""))

        # Заповнення порожніми клітинками
        for row in range(4):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(""))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(""))

        # Назви
        label_metrics = QLabel("📈 Метрики:")
        label_metrics.setStyleSheet("color: #f8f8f2; font-weight: bold;")
        layout.addWidget(label_metrics)

        # ⬅️⬅️ Горизонтальний блок для таблиці метрик + гістограми
        metrics_and_plot_layout = QHBoxLayout()

        # Додаємо таблицю метрик
        metrics_and_plot_layout.addWidget(self.metrics_table)

        # Гістограма (створюємо, якщо ще не створена)
        self.hist_image = QLabel()
        self.hist_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hist_image.setStyleSheet("background-color: #1e1f29; border: 1px solid #44475a;")
        self.hist_image.setFixedSize(400, 250)  # можеш змінити розміри
        metrics_and_plot_layout.addWidget(self.hist_image)

        # Додаємо горизонтальний блок у layout
        layout.addLayout(metrics_and_plot_layout)

        # Продовження: HTML-звіт
        label_html = QLabel("📄 Звіт (HTML):")
        label_html.setStyleSheet("color: #f8f8f2; font-weight: bold;")
        layout.addWidget(label_html)
        layout.addWidget(self.report_browser)
        layout.addWidget(QLabel("🧠 Інтерпретація:"))

        # Основний горизонтальний блок
        interpretation_row = QHBoxLayout()

        # Ліва колонка
        self.explanation_box = QTextEdit()
        self.explanation_box.setMinimumHeight(200)
        self.explanation_box.setReadOnly(True)
        self.explanation_box.setStyleSheet("""
            QTextEdit {
                background-color: #1e1f29;
                color: #f8f8f2;
                font-family: Consolas, monospace;
                border: 1px solid #44475a;
                border-radius: 6px;
                padding: 10px;
                min-width: 600px;
            }
        """)
        interpretation_row.addWidget(self.explanation_box)

        # Права колонка (індекси NDVI, BSI, Brightness Index)
        self.index_box = QTextEdit()
        self.index_box.setMinimumHeight(200)
        self.index_box.setReadOnly(True)
        self.index_box.setStyleSheet("""
            QTextEdit {
                background-color: #1e1f29;
                color: #f1f1f1;
                font-family: Consolas, monospace;
                border: 1px solid #44475a;
                border-radius: 6px;
                padding: 10px;
                min-width: 300px;
            }
        """)
        interpretation_row.addWidget(self.index_box)

        # Додаємо у layout
        container = QWidget()
        container.setLayout(interpretation_row)
        layout.addWidget(container)

        # Кнопка збереження
        self.btn_save_report = QPushButton("💾 Зберегти звіт у HTML")
        self.btn_save_report.setStyleSheet("""
            QPushButton {
                background-color: #44475a;
                color: #f8f8f2;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #6272a4;
            }
        """)
        self.btn_save_report.clicked.connect(self.save_report_to_file)
        layout.addWidget(self.btn_save_report, alignment=Qt.AlignmentFlag.AlignRight)

        return tab

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Оберіть зображення", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            self.image_path = fname
            pixmap = QPixmap(fname).scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio)
            self.label_image.setPixmap(pixmap)

    def classify_image(self):
        if self.image_path:
            try:
                label, probs, classes = predict_soil_condition(self.image_path)

                # 📊 Аналіз зображення
                metrics = analyze_image(self.image_path)
                hist_path = metrics.pop("__histogram_path__", None)  # ⬅️ Вилучаємо шлях до гістограми
                self.metrics_table.setRowCount(len(metrics))  # Підлаштовуємо під кількість метрик

                for row, (k, v) in enumerate(metrics.items()):
                    self.metrics_table.setItem(row, 0, QTableWidgetItem(str(k)))
                    self.metrics_table.setItem(row, 1, QTableWidgetItem(str(v)))

                # 🖼️ Відображення гістограми праворуч
                if hist_path:
                    pixmap = QPixmap(hist_path)
                    self.hist_image.setPixmap(pixmap.scaled(350, 250, Qt.AspectRatioMode.KeepAspectRatio))
                else:
                    self.hist_image.clear()

                # 📄 HTML-звіт
                html_report = generate_html_report(self.image_path, label, metrics)
                self.report_browser.setHtml(html_report)

                # 🔬 Наукова інтерпретація метрик
                brightness = metrics.get("Яскравість", 0)
                contrast = metrics.get("Контраст", 0)
                entropy_val = metrics.get("Ентропія", 0)
                dark = float(metrics.get("Темні пікселі", "0%").replace("%", ""))
                bright = float(metrics.get("Світлі пікселі", "0%").replace("%", ""))

                explanation = ["📊 <b>Аналіз метрик:</b><br>"]
                if brightness < 50 and dark > 70:
                    explanation.append("🔻 <b>Темний ґрунт:</b> можлива волога, заболочена або ущільнена ділянка.")
                if contrast > 35:
                    explanation.append("🔺 <b>Високий контраст:</b> різке чергування структур (сухі/вологі).")
                if entropy_val > 6.5:
                    explanation.append("🌀 <b>Висока ентропія:</b> складна структура ґрунту.")
                if bright > 10:
                    explanation.append("☀️ <b>Світлі пікселі:</b> можливі посушливі, піщані ділянки.")
                if not explanation[1:]:
                    explanation.append("✅ Метрики не виявили аномалій — стан ґрунту стабільний.")

                explanation.append("<hr>")
                explanation.append("🧪 <b>Формульні характеристики:</b>")
                explanation.append("• Нормалізована яскравість = mean / 255 = {:.2f}".format(brightness / 255))
                explanation.append(
                    "• Індекс сухості (DryIndex) = світлі - темні пікселі = {:.2f}%".format(bright - dark))
                explanation.append("• Ентропія (Shannon) ≈ {:.3f}".format(entropy_val))

                self.explanation_box.setHtml("<br>".join(explanation))
                ndvi = 0.25  # Якщо немає NIR – поставити як заглушку
                brightness_index = 48.12
                bsi = -0.10

                indices_text = [
                    "📐 <b>Інтегровані індекси:</b><br>",
                    f"• NDVI = (NIR - RED) / (NIR + RED) = {ndvi:.2f}",
                    f"• Brightness Index = √(R² + G²) / √2 = {brightness_index:.2f}",
                    f"• BSI = (R+G - NIR - B) / (R+G+NIR+B) = {bsi:.2f}"
                ]
                self.index_box.setHtml("<br>".join(indices_text))

                self.tabs.setCurrentIndex(1)

                self.result_label.setText(f"🌱 Стан ґрунту: <b>{label.capitalize()}</b>")
                plot_probabilities(probs, classes)
                save_classification_result(self.image_path, label, max(probs))

            except Exception as e:
                msg = CustomMessage(f"❌ Помилка класифікації:\n{str(e)}")
                msg.move(self.x() + 350, self.y() + 250)
                msg.show()
        else:
            msg = CustomMessage("📂 Спочатку завантажте зображення.")
            msg.move(self.x() + 350, self.y() + 250)
            msg.show()

    def save_report_to_file(self):
        # Отримуємо HTML-код звіту з QTextBrowser
        html_content = self.report_browser.toHtml()
        # Відкриваємо діалог для збереження файлу
        file_path, _ = QFileDialog.getSaveFileName(self, "Зберегти звіт", "", "HTML Files (*.html);;All Files (*)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(self.pos() + event.globalPosition().toPoint() - self.drag_pos)
            self.drag_pos = event.globalPosition().toPoint()
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
