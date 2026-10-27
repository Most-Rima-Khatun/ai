import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.keras.utils.set_random_seed(42)

x = np.random.uniform(-20, 20, 5000).reshape(-1, 1)
y = 5 * (x**4) - 10 * (x**3) - x + 5

scaler_x = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

x_temp, x_test, y_temp, y_test = train_test_split(x_scaled, y_scaled, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.1111, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
val_loss, val_mae = model.evaluate(x_val, y_val, verbose=0)

print("\n--- Results ---")
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Val Loss (MSE): {val_loss:.4f}")
print(f"Val MAE: {val_mae:.4f}\n")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Mean Absolute Error (MAE)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss (MSE)')
plt.legend()

plt.tight_layout()

print("Real X:", scaler_x.inverse_transform(x_test).flatten()[:5])
print("Pred Y:", scaler_y.inverse_transform(model.predict(x_test, verbose=0)).flatten()[:5])

plt.show()
