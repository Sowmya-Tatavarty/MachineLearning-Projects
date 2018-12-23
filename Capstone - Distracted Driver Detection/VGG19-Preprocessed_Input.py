
# coding: utf-8

# ### Imports

# In[ ]:


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

# In[ ]:


with tf.Session() as sess:
  devices = sess.list_devices()
  print(devices)


# In[ ]:


#set the train path
train_path = os.getcwd() + '/imgs/train/'


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

# In[ ]:


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

# In[ ]:


list_of_test_image_urls = glob.glob(os.getcwd() + '/imgs/test/*.*')

first_test_image = plt.imread(list_of_test_image_urls[100])

print(first_test_image.shape)
#list_of_test_image_urls[0]
plt.imshow(first_test_image)
plt.show()


# ### Build a Simple Keras Model

# In[ ]:


import keras


# In[ ]:


from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg19 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# using 100 x 100 X 3 images to train the model within reasonable amount of time and load them in memory
img_height = 150
img_width = 150
img_channels = 3

# get the pretrained VGG19 model from keras application layer
base_model = VGG19(weights='imagenet', include_top=False,input_shape=(img_height,img_width,img_channels))

x = base_model.output
x=Flatten()(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
# set the base model layers to not train
for layer in base_model.layers:
    layer.trainable = False

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)



merged = keras.utils.multi_gpu_model(model,gpus = 4)

merged.summary()


# In[ ]:


batch_size=128
nb_epochs=100


# In[9]:


import keras
opt = Adam(lr = 0.0001, decay = 1e-5)


merged.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

# adding a checkpoint to save the best model during training 
chkptk = keras.callbacks.ModelCheckpoint('VGG19_Another_preprocess.h5',save_best_only = 'True')

train_datagen = ImageDataGenerator(preprocessing_function = keras.applications.vgg19.preprocess_input, brightness_range=[1,2],                 
    validation_split=0.2, rescale = 1/255.) # set validation split #,preprocessing_function = preprocess_input


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
    epochs = nb_epochs, verbose = 1,callbacks = [chkptk,keras.callbacks.EarlyStopping(monitor = 'val_loss',verbose = 1, patience = 5)])
# Save the model
#model.save('VGG16_Another.h5')


# #### The model is learning incremently. It is learning at a steady rate even after increase the learning rate from 0.001 to 0.002. 
# Looks like global minimum is further away and there is atill a lot of learning left. SO increaing the learning rate to 0.005 an rerunning the model

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
    target_size=(150, 150),
    batch_size=batch_size,
    shuffle=True,
    subset='validation') # set as validation data

predictions = model.predict_generator(validation_generator, steps=(validation_generator.samples // batch_size)+1)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())  

score = log_loss(validation_generator.classes, predictions)
print ("log loss score",score)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print (confusion_matrix(validation_generator.classes, predicted_classes))


from sklearn import metrics
report = metrics.classification_report(validation_generator.classes, predicted_classes, target_names=class_labels)
print(report) 


# #### k Fold Cross Validation 

# In[ ]:


from sklearn import metrics
from sklearn.model_selection import KFold


import keras

img_height, img_width, img_channels, batch_size = 150,150,3,11209

opt = Adam(lr = 0.0001, decay = 1e-6)

train_datagen = ImageDataGenerator(rescale=1./255,                          
    validation_split=0.5) # set validation split

validation_generator = train_datagen.flow_from_directory(
    train_path, # same directory as training data
    shuffle=True,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation') # set as validation data


# In[ ]:


from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

model2 = load_model('VGG19_Another_preprocess.h5')


for x, y in validation_generator:
    X_train = []
    Y_train = []
    classes = []
    class_indices = []
    X_train.append(x)
    Y_train.append(y)
    classes.append(validation_generator.classes)
    class_indices.append(validation_generator.class_indices.keys())
    
    
    X_train = X_train[0]
    Y_train = Y_train[0]
    classes = classes[0]
    class_indices = class_indices[0]
    
    assert len(classes) == len(X_train) == len(Y_train), 'Not all of them have equal length'
    
    
    kf=KFold(5)
    y_actual=[]
    y_predict=[]
    fold=0

    oos_y = []
    oos_pred = []
    oos_y_compare = []
    log_loss = []
    accuracy=[]

    for train, test in kf.split(X_train):
        fold+=1
        print("Fold #{}".format(fold))
        x_train=X_train[train]
        y_train=Y_train[train]
        x_test = X_train[test]
        y_test = Y_train[test]
        classes_test = classes[test] 

        # We are only doing predictions for x_test

        pred_softmax = merged.predict(x_test)

        oos_y.append(y_test)
        pred = np.argmax(pred_softmax,axis = 1)
        oos_pred.extend(pred)


        y_compare = np.argmax(y_test,axis = 1)
        oos_y_compare.extend(y_compare)
        score_evaluate = model2.evaluate(x_test,y_test)
        print('Fold Log Loss is {} '.format(score_evaluate[0]))

        log_loss.append(score_evaluate[0])
        accuracy.append(score_evaluate[1])
        print('Fold Score (accuracy) is {}'.format(score_evaluate[1]))


        
        
    break


print('=========================')
print('=========================')


mean_log_loss = np.mean(log_loss)
print('Final mean log loss is {}'.format(mean_log_loss))

mean_accuracy = np.mean(accuracy)
print('Final mean accuracy is {}'.format(mean_accuracy))

print('=========================')
print('=========================')
print('      ')

print('For VGG19, Confusion Matrix is \n {}'.format(confusion_matrix(oos_y_compare, oos_pred)))

print('=========================')
print('=========================')


print('For VGG19, Classification Report is \n {}'.format(classification_report(oos_y_compare, oos_pred, target_names = class_indices)))





# In[ ]:


import submission_function
#Create submission file
submission_function.create_submission_file(150,150,model,"VGG19_New_submission")


# In[ ]:


print('For VGG19, Confusion Matrix is \n {}'.format(confusion_matrix(oos_y_compare, oos_pred)))

print('=========================')
print('=========================')


print('For VGG19, Classification Report is \n {}'.format(classification_report(oos_y_compare, oos_pred, target_names = class_indices)))


# #### Log loss score for  VGG19 model rom the project submission on Kaggle is  1.20704
# 
