#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator  



# In[2]:


cv2.imread("C:/Users/CASPER/Desktop/CV_Project/training/happy/3.jpg").shape


# In[3]:


train = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale= 1/255)


# In[4]:


train_dataset = train.flow_from_directory("C:/Users/CASPER/Desktop/CV_Project/training", 
                                    target_size=(200,200), 
                                    batch_size =3,
                                    class_mode ="binary")

validation_dataset = train.flow_from_directory("C:/Users/CASPER/Desktop/CV_Project/validation", 
                                    target_size=(200,200), 
                                    batch_size =3,
                                    class_mode ="binary")


# In[5]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    #
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    #
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    #
    tf.keras.layers.Flatten(),
    #
    tf.keras.layers.Dense(512, activation='relu'),
    #
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[6]:


model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),  # Use 'learning_rate' instead of 'lr'
    metrics=['accuracy']
)


# In[7]:


model_fit = model.fit(
    train_dataset,
    steps_per_epoch=3,  # Use 'steps_per_epoch' instead of 'step_per_epoch'
    epochs=10,
    validation_data=validation_dataset
)


# In[8]:


validation_dataset.class_indices


# In[9]:


dir_path ='C:/Users/CASPER/Desktop/CV_Project/testing'
for i in os.listdir(dir_path):
    img = load_img(dir_path+'//'+i, target_size=(200,200))  # Use load_img from the correct module
    plt.imshow(img)
    plt.show()

    X = img_to_array(img)  # Use img_to_array from the correct module
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 0:
    
        print("happy")
    else:
        print("unhappy")

