# Deep Neural Network for Malaria Infected Cell Recognition
## AIM
To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
Using data augmentation in the Convolutional Neural Network approach decreases the chances of overfitting. Thus, Malaria detection systems using deep learning proved to be faster than most of the traditional techniques. A Convolutional Neural Network was developed and trained to classify between the parasitized and uninfected smear blood cell images. The classical image features are extracted by CNN which can extract theimage features in three different categories – low-level, mid-level, and high-level features.

## Neural Network Model
![image](https://user-images.githubusercontent.com/75235293/204075948-431f8713-5071-447f-b95c-9881acd02ef9.png)
## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries
### STEP 2:
Read the dataset
### STEP 3:
Create an ImageDataGenerator to flow image data
### STEP 4:
Build the convolutional neural network model and train the model
### STEP 5:
Fit the model
### STEP 6:
Evaluate the model with the testing data
### STEP 7:
Fit the model
### STEP 8:
Plot the performance plot

## PROGRAM
## Developed BY : Manoj Choudhary V 
## Reg no:212221240025
### # Importing Modules
```
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
```
## DataDirectory
```
my_data_dir = '/home/ailab/hdd/dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir + '/test/'
train_path = my_data_dir + '/train/'
os.listdir(train_path)
len(os.listdir(train_path + '/uninfected/'))
len(os.listdir(train_path + '/parasitized/'))
os.listdir(train_path + '/parasitized')[0]
```
## Import and display an image
```
para_img = imread(train_path + '/parasitized/' + os.listdir(train_path + '/parasitized')[0])
para_img.shape
plt.imshow(para_img)
```
## Checking the image dimensions
```
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path + '/uninfected'):
    img = imread(test_path + '/uninfected' + '/' + image_filename)
    d1, d2, colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
sns.jointplot(x=dim1, y=dim2)
```
```
image_shape = (130, 130, 3)
image_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    rescale=1/255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
```
```
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
```
## Create a neural network model
```
model = models.Sequential([
    layers.Input((130, 130, 3)),
    layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, kernel_size=3, activation="relu"),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(32, kernel_size=3, activation="relu"),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", metrics='accuracy', optimizer="adam")
model.summary()
```

## Data generation for training and testing
```
train_image_gen = image_gen.flow_from_directory(train_path, target_size=image_shape[:2], color_mode='rgb',
                                                batch_size=16, class_mode='binary')

train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen

test_image_gen = image_gen.flow_from_directory(test_path, target_size=image_shape[:2], color_mode='rgb',
                                              batch_size=16, class_mode='binary', shuffle=False)

train_image_gen.class_indices
```

##  Train the model
```
results = model.fit(train_image_gen, epochs=5, validation_data=test_image_gen)
model.save('cell_model1.h5')
```
## Visualize the training losses
```
losses = pd.DataFrame(model.history.history)
losses.plot()
```
## Evaluate the model
```
model.evaluate(test_image_gen)
```
## Make predictions and evaluate the model
```
pred_probabilities = model.predict(test_image_gen)
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes, predictions))
confusion_matrix(test_image_gen.classes, predictions)
```
##  Load and process a single image for prediction
```
from tensorflow.keras.preprocessing import image

img = image.load_img('new.png')
img = tf.convert_to_tensor(np.asarray(img))
img = tf.image.resize(img, (130, 130))
img = img.numpy()

type(img)
plt.imshow(img)

x_single_prediction = bool(model.predict(img.reshape(1, 130, 130, 3)) > 0.6)
print(x_single_prediction)

if x_single_prediction == 1:
    print("Uninfected")
else:
    print("Parasitized")
```
## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/manojvenaram/malaria-cell-recognition/assets/94165064/ec42a808-4546-4892-bf5b-cca4ce9dd227)


### Classification Report
![image](https://github.com/manojvenaram/malaria-cell-recognition/assets/94165064/9302870d-4998-4a93-b219-b82f6858acf7)


### Confusion Matrix
![image](https://github.com/manojvenaram/malaria-cell-recognition/assets/94165064/a8b1b1ac-d2c0-429a-a623-1c8ca161245b)

### Single Data prediction
![image](https://github.com/manojvenaram/malaria-cell-recognition/assets/94165064/65deaa0b-5cc2-4c19-9ab2-4a4d769d8145)

## RESULT
Thus, a deep neural network for Malaria infected cell recognized and analyzed the performance .
