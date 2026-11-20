"""
Навчання згорткової нейронної мережі (CNN) на датасеті Fashion MNIST
та побудова графіків точності (Accuracy) та втрат (Loss) відповідно до завдання.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Список класів одягу у Fashion MNIST
CLASS_NAMES = [
    'T-shirt/top',   # 0: Футболка / топ
    'Trouser',       # 1: Штани
    'Pullover',      # 2: Светр
    'Dress',         # 3: Сукня
    'Coat',          # 4: Пальто
    'Sandal',        # 5: Сандалі
    'Shirt',         # 6: Сорочка
    'Sneaker',       # 7: Кросівки
    'Bag',           # 8: Сумка
    'Ankle boot'     # 9: Черевики
]

def load_and_preprocess_data():
    """Завантаження та нормалізація набору даних Fashion MNIST."""
    print("Завантаження даних Fashion MNIST...")
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Нормалізація значень пікселів до діапазону [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Додавання виміру каналу (28x28 -> 28x28x1) для згорткових шарів
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    print(f"Тренувальний набір: {X_train.shape}, мітки: {y_train.shape}")
    print(f"Тестовий набір:     {X_test.shape}, мітки: {y_test.shape}")
    return (X_train, y_train), (X_test, y_test)

def build_model(filter_conv1=32, filter_conv2=64, dense_neurons=128, dropout_rate=0.3):
    """
    Побудова архітектури CNN з використанням різних шарів:
    - Згорткові шари (Conv2D) з різною кількістю фільтрів (32, 64)
    - Шари підвибірки (MaxPooling2D) для зменшення розмірності
    - Шар регуляризації (Dropout) для запобігання перенавчанню
    - Шар вирівнювання (Flatten)
    - Повнозв'язний шар (Dense) з заданою кількістю нейронів
    - Вихідний шар (Dense) з 10 нейронами та активацією softmax
    """
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1), name='input_layer'),
        
        # Блок 1: Згортка + Пулінг
        layers.Conv2D(filter_conv1, (3, 3), activation='relu', name=f'conv2d_{filter_conv1}'),
        layers.MaxPooling2D((2, 2), name='maxpool_1'),
        
        # Блок 2: Згортка з більшою кількістю фільтрів + Пулінг
        layers.Conv2D(filter_conv2, (3, 3), activation='relu', name=f'conv2d_{filter_conv2}'),
        layers.MaxPooling2D((2, 2), name='maxpool_2'),
        
        # Вирівнювання матриці ознак у вектор
        layers.Flatten(name='flatten'),
        
        # Повнозв'язний шар з функцією активації ReLU
        layers.Dense(dense_neurons, activation='relu', name=f'dense_{dense_neurons}'),
        
        # Регуляризація Dropout
        layers.Dropout(dropout_rate, name='dropout'),
        
        # Вихідний шар для 10 класів одягу
        layers.Dense(10, activation='softmax', name='output_layer')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_training_history(history, save_path='plots/accuracy_loss_plot.png'):
    """
    Побудова графіка точності (Accuracy) та похибки (Loss)
    точно у такому стилі та оформленні, як показано на фото завдання.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(11, 4.5))

    # 1. Графік точності (Accuracy over epochs)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train accuracy', color='#1f77b4', linewidth=1.5)
    plt.plot(history.history['val_accuracy'], label='Val accuracy', color='#ff7f0e', linewidth=1.5)
    plt.title('Accuracy over epochs', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(False)

    # 2. Графік похибки (Loss over epochs)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train loss', color='#1f77b4', linewidth=1.5)
    plt.plot(history.history['val_loss'], label='Val loss', color='#ff7f0e', linewidth=1.5)
    plt.title('Loss over epochs', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(loc='upper right', frameon=True)
    plt.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Графік успішно збережено у: {save_path}")
    plt.close()

def main():
    # 1. Завантаження даних
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()

    # 2. Створення моделі
    model = build_model(filter_conv1=32, filter_conv2=64, dense_neurons=128, dropout_rate=0.3)
    model.summary()

    # 3. Навчання моделі (5 епох, як на фото завдання)
    epochs = 5
    batch_size = 64
    print(f"\nПочаток навчання моделі на {epochs} епохах (batch_size={batch_size})...")
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 4. Побудова та збереження графіків (як на фото)
    plot_training_history(history, save_path='plots/accuracy_loss_plot.png')

    # 5. Оцінка моделі на тестовому наборі
    print("\nОцінка моделі на тестових даних...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Тестова точність (Accuracy): {test_acc * 100:.2f}%")
    print(f"Тестова похибка (Loss):      {test_loss:.4f}")

    # 6. Збереження навченої моделі
    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/fashion_cnn_model.keras'
    model.save(model_save_path)
    print(f"Модель успішно збережено у: {model_save_path}")

if __name__ == '__main__':
    main()
