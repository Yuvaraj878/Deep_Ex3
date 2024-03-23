# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Problem Statement: Handwritten Digit Recognition with Convolutional Neural Networks

Objective: Develop a Convolutional Neural Network (CNN) model to accurately classify handwritten digits (0-9) from the MNIST dataset.

Data: The MNIST dataset, a widely used benchmark for image classification, contains grayscale images of handwritten digits (28x28 pixels). Each image is labeled with the corresponding digit (0-9).
## Neural Network Model

![image](./img/diagram.png)


## DESIGN STEPS

### Step1 : Import Libraries:

tensorflow as tf (or tensorflow.keras for a higher-level API)

### Step2 : Load and Preprocess Data:

Use tf.keras.datasets.mnist.load_data() to get training and testing data.

Normalize pixel values (e.g., divide by 255) for better training.
Consider one-hot encoding labels for multi-class classification.

### Step3 : Define Model Architecture:

Use a sequential model (tf.keras.Sequential).

Start with a Convolutional layer (e.g., Conv2D) with filters and kernel size.

Add pooling layers (e.g., MaxPooling2D) for dimensionality reduction.

Repeat Conv2D and MaxPooling for feature extraction (optional).

Flatten the output from the convolutional layers.

Add Dense layers (e.g., Dense) with neurons for classification.

Use appropriate activation functions (e.g., ReLU) and output activation (e.g., softmax for 10 classes).

### Step4 : Compile the Model:

Specify optimizer (e.g., Adam), loss function (e.g., categorical_crossentropy), and metrics (e.g., accuracy).

### Step5 : Train the Model:

Use model.fit(X_train, y_train, epochs=...) to train.

Provide validation data (X_test, y_test) for monitoring performance.

### Step6 : Evaluate the Model:

Use model.evaluate(X_test, y_test) to assess accuracy and other metrics.

## PROGRAM

### Name: YUVARAJ S
### Register Number:212222240119

#### PREPROCESSING
```py
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
```
#### DATA LOADING AND PREPROCESSING
```py
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[5]
single_image.shape
plt.imshow(single_image,cmap='gray')
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[5]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```
#### MODEL ARCHITECTURE
```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(16, (3,3), activation="relu"),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(16, activation="relu"),
    Dense(16, activation='relu'),
    Dense(10, activation="softmax")  
])

model.summary()

model.compile(
    'adam', 
    loss ='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train_scaled ,
          y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot)
        )
metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics.plot()
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.show()

import numpy as np
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print("Name : YUVARAJ.S\nRegister Number : 212222240119\n")
print(confusion_matrix(y_test,x_test_predictions))
print("Name : YUVARAJ.S\nRegister Number : 212222240119\n")
print(classification_report(y_test,x_test_predictions))

```
#### PREDICTION
```py
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

img = image.load_img('4.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor, (28, 28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy() / 255.0

print("Name : YUVARAJ.S\nRegister Number : 212222240119\n")
x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1, 28, 28, 1)), axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()
img_28_gray_inverted = 255.0 - img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy() / 255.0

x_single_prediction_inverted = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1, 28, 28, 1)), axis=1)
print(x_single_prediction_inverted)
plt.imshow(img_28_gray_inverted_scaled.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![download](./img/n1.png)

### Classification Report

![image](./img/n2.png)


### Confusion Matrix

![image](./img/n3.png)


### New Sample Data Prediction

#### OUTPUT:

![](./img/n4.png)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
