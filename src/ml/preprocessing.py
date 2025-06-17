import cv2
import numpy as np
import os
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def preprocess_image(image_path: str, target_size: tuple = (128, 128)) -> np.ndarray:
    """
    Виконує повну передобробку супутникового знімка для подальшої подачі в модель глибинного навчання.

    Етапи обробки:
    1. Перевірка існування зображення.
    2. Завантаження зображення за допомогою OpenCV.
    3. Масштабування (resize) до фіксованого розміру.
    4. Конвертація з BGR (формат OpenCV) до RGB (формат Keras).
    5. Перетворення у формат float32 для сумісності з TensorFlow.
    6. Нормалізація пікселів у діапазон [0, 1].

    :param image_path: Абсолютний або відносний шлях до вхідного зображення.
    :param target_size: Очікуваний розмір на виході у вигляді кортежу (ширина, висота).
    :return: Нормалізоване RGB-зображення у вигляді масиву NumPy з формою (висота, ширина, канали).
    :raises FileNotFoundError: Якщо файл зображення не знайдено.
    :raises ValueError: Якщо зображення не вдалося зчитати.
    """

    if not os.path.exists(image_path):
        logging.error(f"Файл не знайдено: {image_path}")
        raise FileNotFoundError(f"Файл не знайдено: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Зображення не вдалося зчитати: {image_path}")
        raise ValueError(f"Зображення не вдалося зчитати: {image_path}")

    original_shape = image.shape
    logging.info(f"Зчитано зображення {image_path} з початковою формою {original_shape}")

    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    logging.info(f"Зображення масштабовано до {target_size}")

    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    logging.debug(f"Конвертація з BGR до RGB виконана")

    image_normalized = image_rgb.astype(np.float32) / 255.0
    logging.debug(f"Нормалізація пікселів завершена (діапазон: {image_normalized.min()}–{image_normalized.max()})")

    return image_normalized
