import os
import cv2
import numpy as np
import logging
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def load_images_from_folder(folder_path: str, target_size=(64, 64)) -> tuple[np.ndarray, np.ndarray]:
    """
    Завантажує та обробляє зображення з підпапок (одна папка — один клас).

    :param folder_path: Абсолютний шлях до папки з класами.
    :param target_size: Розмір для зміни масштабу зображень.
    :return: Кортеж (X, y), де X — масив зображень, y — мітки класів.
    """
    X, y = [], []

    for label in os.listdir(folder_path):
        class_dir = os.path.join(folder_path, label)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            img_path = os.path.join(class_dir, fname)

            if not os.path.isfile(img_path):
                logging.warning(f"Пропущено не-файл: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Не вдалося зчитати: {img_path}")
                continue

            img = cv2.resize(img, target_size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(gray.flatten())
            y.append(label)

    return np.array(X), np.array(y)


def train_svm_classifier(train_dir: str, model_path: str = "models/svm_soil_model.pkl"):
    """
    Навчає SVM-модель на основі зображень із підпапок і зберігає її у файл.

    :param train_dir: Шлях до директорії train (кожна підпапка — клас).
    :param model_path: Шлях для збереження моделі.
    """
    logging.info("Завантаження даних...")
    X, y = load_images_from_folder(train_dir)

    if X.size == 0 or y.size == 0:
        logging.error("❌ Дані не знайдено. Перевір, чи є зображення в підпапках.")
        return

    logging.info(f"✅ Кількість зразків: {len(X)}, Розмірність ознак: {X.shape[1]}")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    logging.info("Навчання моделі SVM...")
    model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=label_encoder.transform(label_encoder.classes_),
        target_names=label_encoder.classes_,
        zero_division=0
    )
    print("\n=== Класифікаційний звіт ===")
    print(report)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump((model, label_encoder), model_path)
    logging.info(f"✅ Модель збережено у: {model_path}")


if __name__ == "__main__":
    # Побудова абсолютного шляху до data/raw/train
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    train_path = os.path.join(root, "data", "raw", "train")

    logging.info(f"Запуск з: {train_path}")
    train_svm_classifier(train_path)
