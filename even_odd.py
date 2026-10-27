import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist   
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((3, 3), strides=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((3, 3), strides=(1, 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.15, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

# ১ম গ্রাফ (Loss & Accuracy)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.show()

# c) Display the numbers
plt.figure(figsize=(10, 2))  # <--- এই লাইনটি যোগ করো
indices = np.random.choice(len(x_test), 5, replace=False)
for i, idx in enumerate(indices):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {np.argmax(model.predict(x_test[idx:idx+1], verbose=0))}")
    plt.axis('off')
plt.show()

# d) Display Even numbers only
plt.figure(figsize=(10, 2))  # <--- এই লাইনটি যোগ করো
even_idx = np.where(y_test % 2 == 0)[0]
indices = np.random.choice(even_idx, 5, replace=False)
for i, idx in enumerate(indices):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {np.argmax(model.predict(x_test[idx:idx+1], verbose=0))}")
    plt.axis('off')
plt.show()

# e) Display Odd numbers only
plt.figure(figsize=(10, 2))  # <--- এই লাইনটি যোগ করো
odd_idx = np.where(y_test % 2 != 0)[0]
indices = np.random.choice(odd_idx, 5, replace=False)
for i, idx in enumerate(indices):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {np.argmax(model.predict(x_test[idx:idx+1], verbose=0))}")
    plt.axis('off')
plt.show()
