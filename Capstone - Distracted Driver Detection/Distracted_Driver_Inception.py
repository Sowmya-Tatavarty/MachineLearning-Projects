
# coding: utf-8

# ### Imports

# In[11]:


import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,                                        ZeroPadding2D

from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
#import cv2
import math
import pickle
import datetime
import pandas as pd
import glob

import os
import tensorflow as tf


# ### Read CSV into Notebook

# In[12]:


with tf.Session() as sess:
  devices = sess.list_devices()
  print(devices)


# In[13]:


csv_path = os.getcwd() + '/driver_imgs_list.csv'
d_csv = pd.read_csv(csv_path)


# ### Class Definitions
# 
# The 10 classes to predict are:  
# 
# c0: safe driving  
# c1: texting - right  
# c2: talking on the phone - right  
# c3: texting - left  
# c4: talking on the phone - left  
# c5: operating the radio  
# c6: drinking  
# c7: reaching behind  
# c8: hair and makeup  
# c9: talking to passenger  

# In[14]:


class_dict = {
    'c0': "safe driving",
    'c1': "texting - right",
    'c2': "talking on the phone - right",
    'c3': "texting - left",
    'c4': "talking on the phone - left ",
    'c5': "operating the radio ",
    'c6': "drinking",
    'c7': "reaching behind",
    'c8': "hair and makeup ",
    'c9': "talking to passenger ",
} 


# ### Read in Images

# In[15]:


list_of_test_image_urls = glob.glob(os.getcwd() + '/imgs/test/*.*')

first_test_image = plt.imread(list_of_test_image_urls[100])

print(first_test_image.shape)
#list_of_test_image_urls[0]
plt.imshow(first_test_image)
plt.show()


# ### Build a Simple Keras Model

# In[16]:


train_path = os.getcwd() + '/imgs/train/'


# In[17]:


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Activation, BatchNormalization
from keras import backend as K
import keras

img_height = 150
img_width = 150
img_channels = 3

base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(img_height,img_width,img_channels))
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)



model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


merged = keras.utils.multi_gpu_model(model,gpus = 4)

merged.summary()


# In[20]:


batch_size=32
nb_epochs=5


# In[ ]:


import keras
merged.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001), metrics=['accuracy'])

chkptk = keras.callbacks.ModelCheckpoint('Inception_best_Frozen_Pre.h5',save_best_only = 'True')



train_datagen = ImageDataGenerator(brightness_range=[1,2],
    validation_split=0.2,preprocessing_function = keras.applications.inception_v3.preprocess_input) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_path, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation') # set as validation data

history=merged.fit_generator(
    train_generator,
    #steps_per_epoch = train_generator.samples // batch_size,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps=validation_generator.samples // batch_size,
    epochs = nb_epochs, verbose = 1,callbacks = [chkptk,keras.callbacks.EarlyStopping(monitor = 'val_loss',verbose = 1, patience = 10)])
#model.save('VGG16_Another.h5')


# In[9]:


nb_epochs=500
for layer in model.layers[:175]:
   layer.trainable = False
for layer in model.layers[175:]:
   layer.trainable = True


# In[ ]:


sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

merged.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

chkptk = keras.callbacks.ModelCheckpoint('Resnet_best_UnFrozen_Pre.h5',save_best_only = 'True')

history2=merged.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps=validation_generator.samples // batch_size,
    epochs = nb_epochs, verbose = 1,callbacks = [chkptk,keras.callbacks.EarlyStopping(monitor = 'val_loss',verbose = 1, patience = 10)])
# Save the model


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()


# In[ ]:


import numpy as np
from sklearn.metrics import log_loss
import keras
batch_size=32

train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.2) # set validation split
validation_generator = train_datagen.flow_from_directory(
    train_path, # same directory as training data
    target_size=(100, 100),
    batch_size=batch_size,
    shuffle=False,
    subset='validation') # set as validation data

# Evaluate your model.
scores = model.evaluate_generator(validation_generator, steps=validation_generator.samples // batch_size)
print("Loss: " + str(scores[0]) + " Accuracy: " + str(scores[1]))

predictions = model.predict_generator(validation_generator, steps=(validation_generator.samples // batch_size)+1)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())  

score = log_loss(validation_generator.classes, predictions)
print ("log loss score",score)


# Create sample submission

# In[ ]:


import submission_function
#Create submission file
submission_function.create_submission_file(100,100,model,"Inception_submission")


# #### Log loss score for Inception model from the project submission on Kaggle is 2.23533
