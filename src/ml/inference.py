import os
from src.config import USE_KERAS_MODEL, KERAS_MODEL_PATH, SVM_MODEL_PATH
from src.ml.model_utils import predict_with_keras_model, predict_with_svm_model



def predict_soil_condition(image_path: str):
    """
    Класифікує стан ґрунту, використовуючи вибрану модель (SVM або Keras).
    :param image_path: Шлях до зображення
    :return: (predicted_label, probabilities, class_names)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Зображення не знайдено: {image_path}")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    if USE_KERAS_MODEL:
        model_path = os.path.join(root_dir, KERAS_MODEL_PATH)
        return predict_with_keras_model(image_path, model_path)
    else:
        model_path = os.path.join(root_dir, SVM_MODEL_PATH)
        return predict_with_svm_model(image_path, model_path)


# 🧪 Локальний тест
if __name__ == "__main__":
    test_image = "data/raw/train/wet/wet_1.jpg"
    try:
        label, probs, classes = predict_soil_condition(test_image)
        print(f"✅ Результат: {label}")
        for c, p in zip(classes, probs):
            print(f"{c}: {p:.2f}")
    except Exception as e:
        print(str(e))
