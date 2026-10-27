import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10

tf.keras.utils.set_random_seed(42)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

train_mask = np.isin(y_train, [0, 1, 9])
test_mask = np.isin(y_test, [0, 1, 9])

X_train, y_train = X_train[train_mask], y_train[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]

y_train[y_train == 9] = 2
y_test[y_test == 9] = 2

# ==========================================
# RAM বাঁচাতে ডেটার পরিমাণ কমানো হলো
# ==========================================
X_train, y_train = X_train[:3000], y_train[:3000]
X_test, y_test = X_test[:500], y_test[:500]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

IMG_SIZE = 96
X_train = tf.image.resize(X_train, (IMG_SIZE, IMG_SIZE)).numpy() / 255.0
X_val = tf.image.resize(X_val, (IMG_SIZE, IMG_SIZE)).numpy() / 255.0
X_test = tf.image.resize(X_test, (IMG_SIZE, IMG_SIZE)).numpy() / 255.0

base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

print("Training Phase 1...")
history1 = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32, callbacks=callbacks, verbose=1)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training Phase 2 (Fine-Tuning)...")
history2 = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=32, callbacks=callbacks, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

print("\n--- Results ---")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Val Loss: {val_loss:.4f}")
print(f"Val Accuracy: {val_acc*100:.2f}%\n")

acc = history1.history['accuracy'] + history2.history['accuracy']
v_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
v_loss = history1.history['val_loss'] + history2.history['val_loss']
phase_line = len(history1.history['accuracy']) - 0.5

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train')
plt.plot(v_acc, label='Val')
plt.axvline(x=phase_line, color='red', linestyle='--')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train')
plt.plot(v_loss, label='Val')
plt.axvline(x=phase_line, color='red', linestyle='--')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()
