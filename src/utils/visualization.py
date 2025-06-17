import matplotlib.pyplot as plt

def plot_probabilities(probabilities: list, class_names: list):
    """
    Виводить bar-графік для ймовірностей класифікації.
    :param probabilities: список ймовірностей для кожного класу.
    :param class_names: список назв класів.
    """
    plt.figure(figsize=(8, 5))
    bars = plt.bar(class_names, probabilities, color='skyblue')
    plt.title("Ймовірності класифікації")
    plt.ylabel("Ймовірність")
    plt.xlabel("Клас ґрунту")

    # Додати значення поверх стовпчиків
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{prob:.2f}",
                 ha='center', va='bottom', fontsize=10)

    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.show()
