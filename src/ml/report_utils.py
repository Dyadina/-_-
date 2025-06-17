import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
from scipy.stats import entropy as scipy_entropy
from tempfile import NamedTemporaryFile


def analyze_image(image_path: str) -> dict:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("❌ Не вдалося зчитати зображення")

    # Базові метрики
    mean = np.mean(image)
    std = np.std(image)
    dark_pixels = np.sum(image < 50) / image.size * 100
    bright_pixels = np.sum(image > 200) / image.size * 100

    # Ентропія
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()
    img_entropy = scipy_entropy(hist_norm, base=2)

    # Побудова гістограми яскравості
    hist_img_path = None
    with NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.figure(figsize=(5, 3))
        plt.hist(image.ravel(), bins=256, range=(0, 256), color='skyblue', alpha=0.7)
        plt.title("Гістограма яскравості")
        plt.xlabel("Яскравість")
        plt.ylabel("Кількість пікселів")
        plt.tight_layout()
        plt.savefig(tmpfile.name)
        hist_img_path = tmpfile.name
        plt.close()

    return {
        "Яскравість": round(mean, 2),
        "Контраст": round(std, 2),
        "Ентропія": round(img_entropy, 3),
        "Темні пікселі": f"{dark_pixels:.2f}%",
        "Світлі пікселі": f"{bright_pixels:.2f}%",
        "__histogram_path__": hist_img_path  # технічне поле для GUI
    }


def generate_report(label: str, metrics: dict) -> str:
    lines = [
        f"📌 Клас: {label.capitalize()}",
        "🔍 Параметри зображення:",
        *(f"• {k}: {v}" for k, v in metrics.items()),
        "",
        "📘 Висновок:",
        f"Стан ґрунту класифіковано як '{label}', виходячи з аналізу зображення."
    ]
    return "\n".join(lines)


def generate_html_report(image_path: str, label: str, metrics: dict) -> str:
    """
    Генерує HTML-звіт зі зображенням, метриками і висновком.
    """
    # Закодувати зображення в base64
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ background-color: #1e1f29; color: #f8f8f2; font-family: Segoe UI, sans-serif; padding: 20px; }}
            h1 {{ color: #50fa7b; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            td, th {{ border: 1px solid #6272a4; padding: 10px; }}
            th {{ background-color: #282a36; color: #ff79c6; }}
            img {{ margin-top: 20px; max-width: 600px; border-radius: 10px; }}
            .footer {{ margin-top: 30px; font-style: italic; }}
        </style>
    </head>
    <body>
        <h1>Звіт про класифікацію ґрунту</h1>
        <h2>🌱 Результат: {label.capitalize()}</h2>

        <table>
            <tr><th>Метрика</th><th>Значення</th></tr>
            {''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in metrics.items())}
        </table>

        <div class="footer">
            <p>📌 Зображення, яке аналізувалося:</p>
            <img src="data:image/jpeg;base64,{encoded_image}" />
            <p>Звіт згенеровано автоматично.</p>
        </div>
    </body>
    </html>
    """
    return html
