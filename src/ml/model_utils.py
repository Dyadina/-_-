import os
import cv2
import numpy as np
import joblib
from src.ml.preprocessing import preprocess_image


def predict_with_keras_model(image_path: str, model_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Зображення не знайдено: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Модель не знайдено: {model_path}")

    # ❗ Імпортуємо TensorFlow тільки тут
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    image = preprocess_image(image_path, target_size=(128, 128))
    image_batch = np.expand_dims(image, axis=0)

    predictions = model.predict(image_batch)[0]
    class_idx = np.argmax(predictions)
    class_names = list(model.class_names) if hasattr(model, "class_names") else ["drought", "erosion", "healthy", "wet"]

    predicted_label = class_names[class_idx]
    return predicted_label, predictions.tolist(), class_names


def predict_with_svm_model(image_path: str, model_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Зображення не знайдено: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Модель не знайдено: {model_path}")

    model, label_encoder = joblib.load(model_path)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Не вдалося зчитати зображення: {image_path}")

    resized = cv2.resize(image, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten().reshape(1, -1)

    predicted_class_idx = model.predict(flat)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(flat)[0]
    else:
        probabilities = [1.0 if i == predicted_class_idx else 0.0 for i in range(len(label_encoder.classes_))]

    return predicted_label, probabilities.tolist(), label_encoder.classes_.tolist()
