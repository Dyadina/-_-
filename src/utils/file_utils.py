import csv
import os
from datetime import datetime


def save_classification_result(image_path: str, label: str, probability: float, csv_path: str = "results.csv"):
    """
    Зберігає результат класифікації в CSV.
    :param image_path: шлях до зображення
    :param label: передбачений клас
    :param probability: найбільша ймовірність (від 0 до 1)
    :param csv_path: шлях до CSV-файлу
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Дата", "Зображення", "Клас", "Ймовірність"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_path,
            label,
            f"{probability:.4f}"
        ])
