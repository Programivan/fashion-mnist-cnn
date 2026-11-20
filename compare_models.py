"""
Порівняння різних архітектур нейронних мереж для Fashion MNIST:
1. Повнозв'язна мережа (MLP): Dense шари (128, 64 нейрони)
2. Базова згорткова мережа (CNN Standard): Conv2D(32), Conv2D(64), Dense(128)
3. Поглиблена згорткова мережа (CNN Deep): Conv2D(32, 64, 128), BatchNorm, Dense(256)
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def load_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = (X_train.astype(np.float32) / 255.0).reshape(-1, 28, 28, 1)
    X_test = (X_test.astype(np.float32) / 255.0).reshape(-1, 28, 28, 1)
    return (X_train, y_train), (X_test, y_test)

def build_mlp():
    """Модель 1: Повнозв'язна мережа (Dense шари: 128 та 64 нейрони)."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', name='dense_128'),
        layers.Dense(64, activation='relu', name='dense_64'),
        layers.Dense(10, activation='softmax', name='output')
    ], name="MLP_Dense_128_64")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_standard():
    """Модель 2: Базова згорткова мережа (Фільтри: 32, 64; Dense: 128 нейронів)."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', name='conv_32'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', name='conv_64'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', name='dense_128'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax', name='output')
    ], name="CNN_Standard_32_64")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_deep():
    """Модель 3: Поглиблена мережа з BatchNorm (Фільтри: 32, 64, 128; Dense: 256 нейронів)."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv_32'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_64'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', name='conv_128'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu', name='dense_256'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax', name='output')
    ], name="CNN_Deep_BatchNorm_256")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    (X_train, y_train), (X_test, y_test) = load_data()

    models_dict = {
        "1. MLP (Dense 128+64)": build_mlp(),
        "2. CNN Standard (32+64, Dense 128)": build_cnn_standard(),
        "3. CNN Deep (32+64+128, Dense 256)": build_cnn_deep()
    }

    results = {}
    epochs = 5
    batch_size = 64

    for name, model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Тренування моделі: {name}")
        print(f"{'='*60}")
        model.summary()

        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        train_time = time.time() - start_time

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        param_count = model.count_params()

        results[name] = {
            'history': history.history,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'train_time': train_time,
            'param_count': param_count
        }

    # Підсумкова таблиця
    print(f"\n{'='*75}")
    print(f"{'Модель':<36} | {'Параметри':<10} | {'Test Acc':<9} | {'Test Loss':<9} | {'Час (с)':<7}")
    print(f"{'-'*75}")
    for name, data in results.items():
        print(f"{name:<36} | {data['param_count']:<10,} | {data['test_acc']*100:6.2f}%  | {data['test_loss']:<9.4f} | {data['train_time']:5.1f}s")
    print(f"{'='*75}")

    # Побудова графіка порівняння
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(14, 5))

    # Графік валідаційної точності по епохах
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(range(1, epochs + 1), data['history']['val_accuracy'], marker='o', label=f"{name} ({data['test_acc']*100:.1f}%)")
    plt.title('Порівняння валідаційної точності (Val Accuracy)', fontsize=12)
    plt.xlabel('Епоха', fontsize=10)
    plt.ylabel('Точність (Accuracy)', fontsize=10)
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Стовпчикова діаграма підсумкової точності на тестовій вибірці
    plt.subplot(1, 2, 2)
    model_names = [k.split(' (')[0] for k in results.keys()]
    accs = [v['test_acc'] * 100 for v in results.values()]
    colors = ['#4285F4', '#34A853', '#EA4335']
    bars = plt.bar(model_names, accs, color=colors, width=0.55)
    plt.title('Підсумкова тестова точність (%)', fontsize=12)
    plt.ylabel('Точність (%)', fontsize=10)
    plt.ylim([80, 95])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('plots/models_comparison.png', dpi=300)
    print("\nГрафік порівняння збережено у: plots/models_comparison.png")
    plt.close()

if __name__ == '__main__':
    main()
