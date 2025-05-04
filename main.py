import keras.regularizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import f1_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Load data from CSV file
data = pd.read_csv('train_cropd2.csv')
val = pd.read_csv('val_cropedd.csv')
test = pd.read_csv('test_output2.csv')

# Extract the image paths and labels from the data
image_paths = data['path'].values
labels = data['label'].values
image_paths_val = val['path'].values
labels_val = val['label'].values
image_paths_test = test['path'].values
labels_test = test['label'].values

 # Define the number of classes
classes = np.unique(labels).shape[0]
classes_val = np.unique(labels_val).shape[0]
classes_test = np.unique(labels_test).shape[0]

# Initialize lists to store images and labels
data_images = []
data_labels = []
data_images_val = []
data_labels_val = []
data_images_test = []
data_labels_test = []

# Load and preprocess the images train
for image_path, label in zip(image_paths, labels):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
      # Check if image loading was successful
    image = cv2.resize(image, (30, 30))
    data_images.append(image)
    data_labels.append(label)

# Load and preprocess the images val
for image_paths_val, labels_val in zip(image_paths_val, labels_val):
    image = cv2.imread(image_paths_val, cv2.IMREAD_COLOR)
      # Check if image loading was successful
    image = cv2.resize(image, (30, 30))
    data_images_val.append(image)
    data_labels_val.append(labels_val)

# Load and preprocess the images test
for image_paths_test, labels_test in zip(image_paths_test, labels_test):
        image = cv2.imread(image_paths_test, cv2.IMREAD_COLOR)
        # Check if image loading was successful
        image = cv2.resize(image, (30, 30))
        data_images_test.append(image)
        data_labels_test.append(labels_test)

# Convert lists into numpy arrays
data_images = np.array(data_images)
data_labels = np.array(data_labels)
data_images_val = np.array(data_images_val)
data_labels_val = np.array(data_labels_val)
data_images_test = np.array(data_images_test)
data_labels_test = np.array(data_labels_test)
print("Train Shape::", data_images.shape, data_labels.shape)
print("val Shape::", data_images_val.shape, data_labels_val.shape)
print("Test Shape::", data_images_test.shape, data_labels_test.shape)



# Convert the labels into one-hot encoding
y_train = to_categorical(data_labels, classes)
y_val = to_categorical(data_labels_val, classes_val)
y_test = to_categorical(data_labels_test, classes_test)


data_images = data_images/255
data_images_val = data_images_val/255
data_images_test = data_images_test/255


# Build the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=data_images.shape[1:]))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dropout(rate=0.2))
model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dropout(rate=0.2))
model.add(Dense(classes, activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 20
history = model.fit(data_images, y_train, batch_size=32, epochs=epochs, validation_data=(data_images_val, y_val))




# Plot accuracy and loss graphs
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Evaluate the model on the test data
loss, acc = model.evaluate(data_images_test, y_test)
print("Loss:", loss)
print("Accuracy:", acc)
# Save the model
model.save('traffic_classifier1.h5')

