"""
Візуалізація передбачень натренованої нейромережі на тестових зображеннях Fashion MNIST.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    'T-shirt/top',   # 0
    'Trouser',       # 1
    'Pullover',      # 2
    'Dress',         # 3
    'Coat',          # 4
    'Sandal',        # 5
    'Shirt',         # 6
    'Sneaker',       # 7
    'Bag',           # 8
    'Ankle boot'     # 9
]

def main():
    model_path = 'models/fashion_cnn_model.keras'
    if not os.path.exists(model_path):
        print(f"Помилка: Модель {model_path} не знайдена. Спочатку запустіть train.py!")
        return

    print("Завантаження навченої моделі...")
    model = tf.keras.models.load_model(model_path)

    print("Завантаження тестових даних...")
    (_, _), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_test_norm = (X_test.astype(np.float32) / 255.0).reshape(-1, 28, 28, 1)

    # Вибір 15 випадкових зразків
    np.random.seed(42)
    indices = np.random.choice(len(X_test), 15, replace=False)
    sample_images = X_test_norm[indices]
    sample_labels = y_test[indices]

    predictions = model.predict(sample_images)

    # Відображення сітки 3x5
    plt.figure(figsize=(15, 9))
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        plt.imshow(X_test[indices[i]], cmap='gray')
        
        pred_label = np.argmax(predictions[i])
        confidence = np.max(predictions[i]) * 100
        true_label = sample_labels[i]

        is_correct = (pred_label == true_label)
        color = 'darkgreen' if is_correct else 'crimson'
        
        title_text = f"True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]} ({confidence:.1f}%)"
        plt.title(title_text, color=color, fontsize=10, fontweight='bold')
        plt.axis('off')

    plt.suptitle("Приклади класифікації тестових зображень Fashion MNIST", fontsize=15, y=0.98)
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    save_path = 'plots/sample_predictions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Візуалізацію передбачень збережено у: {save_path}")
    plt.close()

if __name__ == '__main__':
    main()
