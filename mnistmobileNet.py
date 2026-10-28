import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Resizing,GlobalAveragePooling2D,Dense,Rescaling
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()

for i in range(10):
 plt.subplot(2,5,i+1)
 plt.imshow(X_train[i].squeeze(),cmap='gray')
 plt.title(f"samples: {y_train[i]}")
 plt.axis('off')
plt.tight_layout()
plt.show()

X_train=X_train.astype("float32")/255.0
X_test=X_test.astype("float32")/255.0

X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)

#add 1 to 3 channel
X_train=np.repeat(X_train,3,axis=-1)
X_test=np.repeat(X_test,3,axis=-1)

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,shuffle=True,test_size=0.10)

base_model=MobileNet(
weights='imagenet',
include_top=False,
input_shape=(96,96,3)
 )
base_model.trainable=True
for layer in base_model.layers[:-20]:
 layers.trainable=False

model=Sequential([
Input(shape=(28,28,3)),
Resizing(96,96),
Rescaling(scale=2.0,offset=-1),
base_model,
GlobalAveragePooling2D(),
Dense(10,activation='softmax')

]
)
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=5)

test_loss,test_accuracy=model.evaluate(X_test,y_test)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.title("Loss graph")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.title("accuracy graph")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.tight_layout()
plt.show()

predictions=model.predict(X_test)
predicted_digit=predictions.argmax(axis=1)

for i in range(10):
 plt.subplot(2,5,i+1)
 plt.imshow(X_test[i].squeeze(),cmap='gray')
 plt.title(f"predicted classes: {predicted_digit[i]}\nActual classes: {y_test[i]}")
 plt.axis('off')
plt.tight_layout()
plt.show()