import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense , Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics

def polynomial(x):
	return 5*x**3 - 10*x**2 - 20*x + 10

np.random.seed(4)

x_samples = np.random.uniform(-20,20,1000)
y_samples = polynomial(x_samples)

x_normalized = (x_samples - np.min(x_samples)) / (np.max(x_samples) - np.min(x_samples))
y_normalized = (y_samples - np.min(y_samples)) / (np.max(y_samples) - np.min(y_samples))

test_size = int(0.10 * len(x_samples))
val_size = int(0.10 * len(x_samples))
train_size = int(len(x_samples)) - test_size - val_size

x_train,x_val,x_test = (
	x_normalized[:train_size],
	x_normalized[train_size : train_size + val_size],
	x_normalized[train_size + val_size :])

y_train,y_val,y_test = (y_normalized[:train_size],y_normalized[train_size : train_size + val_size],y_normalized[train_size + val_size :])

inputs = Input((1,), name = 'Inputlayer')

x = Dense(16, activation = 'relu', name = 'Hiddenlayer1')(inputs)
x = Dense(32, activation = 'relu', name = 'Hiddenlayer2')(x)
x = Dense(64, activation = 'relu', name = 'Hiddenlayer3')(x)
x = Dense(128, activation = 'relu', name = 'Hiddenlayer4')(x)

outputs = Dense(1, name = 'Outputlayer')(x)
model = Model(inputs,outputs)
model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),loss = 'mean_squared_error',metrics = [metrics.R2Score(name = 'accuracy')])


early_stopping = EarlyStopping(monitor = 'val_loss',patience = 3,restore_best_weights = True)


history = model.fit(
	x_train,y_train,
	epochs=10,
	validation_data = (x_val,y_val),
	batch_size = 32,
	callbacks = [early_stopping],
	verbose = 1)


plt.figure(figsize = (10,5))
plt.subplot(2,2,1)
plt.plot(history.history['accuracy'],label = 'train accuracy')
plt.plot(history.history['val_accuracy'],label = 'test accuracy')
plt.title('Training vs testing accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize = (10,5))
plt.subplot(2,2,2)
plt.plot(history.history['loss'],label = 'Training Loss')
plt.plot(history.history['val_loss'],label='Test loss')
plt.title('Training vs testing loss graph')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


model.evaluate(x_test,y_test)
predictions = model.predict(x_test)

plt.subplot(2,2,3)
plt.scatter(x_test,y_test,marker = 'o', label = 'ground truth')
plt.scatter(x_test,predictions,marker = '+' , label = 'prediction accuracy')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
