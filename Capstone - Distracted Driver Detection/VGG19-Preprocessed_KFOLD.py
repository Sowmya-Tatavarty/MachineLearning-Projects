
# coding: utf-8

# ### Imports

# In[22]:


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

# In[23]:


with tf.Session() as sess:
  devices = sess.list_devices()
  print(devices)


# In[24]:


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

# In[25]:


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

# In[26]:


import cv2


# In[27]:


list_of_test_image_urls = glob.glob(os.getcwd() + '/imgs/test/*.*')

first_test_image = plt.imread(list_of_test_image_urls[100])

print(first_test_image.shape)
#list_of_test_image_urls[0]
plt.imshow(first_test_image)
plt.show()


# ### Build a Simple Keras Model

# In[28]:


import keras


# In[29]:


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


# In[30]:


batch_size2 = 128
batch_size=10000
nb_epochs=100


# In[31]:


import keras
opt = Adam(lr = 0.0001, decay = 1e-5)


merged.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

# adding a checkpoint to save the best model during training 
chkptk = keras.callbacks.ModelCheckpoint('VGG19_Kfold.h5',save_best_only = 'True')

train_datagen = ImageDataGenerator(preprocessing_function = keras.applications.vgg19.preprocess_input, brightness_range=[1,2],                 
    validation_split=0.2, rescale = 1/255.) # set validation split #,preprocessing_function = preprocess_input


train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=17943,
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_path, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=4481,
    subset='validation') # set as validation data


#model.save('VGG16_Another.h5')


# In[33]:


from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

from sklearn import metrics
from sklearn.model_selection import KFold


import keras




for x, y in train_generator:
    X_train = []
    Y_train = []
    classes = []
    class_indices = []
    X_train.append(x)
    Y_train.append(y)
    classes.append(train_generator.classes)
    class_indices.append(train_generator.class_indices.keys())
    
    print('At Step 1')
    
    
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
    
    print('At Step 2')

    for train, test in kf.split(X_train):
        fold+=1
        print("Fold #{}".format(fold))
        x_train=X_train[train]
        y_train=Y_train[train]
        x_test = X_train[test]
        y_test = Y_train[test]
        classes_test = classes[test] 

        # We are only doing predictions for x_test
        
        history=merged.fit(
        x_train,y_train, batch_size = batch_size2, 
        validation_data = (x_test,y_test), 
        epochs = nb_epochs, verbose = 1,callbacks = [chkptk,keras.callbacks.EarlyStopping(monitor = 'val_loss',verbose = 1, patience = 5)])

        pred_softmax = merged.predict(x_test)

        oos_y.append(y_test)
        pred = np.argmax(pred_softmax,axis = 1)
        oos_pred.extend(pred)


        y_compare = np.argmax(y_test,axis = 1)
        oos_y_compare.extend(y_compare)
        score_evaluate = merged.evaluate(x_test,y_test)
        print('Fold Log Loss is {} '.format(score_evaluate[0]))

        log_loss.append(score_evaluate[0])
        accuracy.append(score_evaluate[1])
        print('Fold Score (accuracy) is {}'.format(score_evaluate[1]))
        
    print('Done')
    break


# In[15]:


from sklearn.metrics import classification_report, confusion_matrix
print (confusion_matrix(validation_generator.classes, predicted_classes))


from sklearn import metrics
report = metrics.classification_report(validation_generator.classes, predicted_classes, target_names=class_labels)
print(report) 


# In[15]:


predictions_kaggle = []
import parmap


import pandas as pd
import os
import keras
import numpy
from sklearn.metrics import log_loss
import glob
import matplotlib.pyplot as plt
import parmap
import cv2
import numpy as np
from keras.applications.vgg19 import preprocess_input

# In[ ]:


def getmatrix2(x):
    var = cv2.resize(plt.imread(x),(150,150))
    return var

def create_submission_file(img_height,img_width,model,sub_file_name):

    test_globs = glob.glob(os.getcwd() + '/imgs/test/*.*')
    file_names = [x.split('/')[-1] for x in test_globs]
    batch_size1 = 10000

    # predict class probabilities on test data
    for i in range(0,len(test_globs),batch_size1):
        print('Processing {} to {} and length of predictions before append is {}'.format(i,i+batch_size1,len(predictions_kaggle)))
        X_test = []
        X_test.append(parmap.map(getmatrix2,test_globs[i:i+batch_size1],pm_chunksize = 100))
        X_test = np.array(X_test[0])
        Y_test = []
        for i in range(len(X_test)):
            Y_test.append(1)

        from keras.applications.vgg19 import preprocess_input

        train_datagen = ImageDataGenerator(rescale = 1./255,preprocessing_function = preprocess_input, brightness_range=[1,2]) # set validation split

        train_generator = train_datagen.flow(
            X_test,Y_test,
            shuffle=False,
            batch_size=len(X_test)) # set as training data

        X_test2 = []



        for x,_ in train_generator:
            print(len(X_test2))
            X_test2.extend(x)
            break
            
            
#        predictions_kaggle.extend(model.predict_generator(train_generator,10000 // 32 + 1 , verbose = 1))

        for image in X_test2:
            predictions_kaggle.append(model.predict(image.reshape(-1,img_height,img_width,3))[0])
            
            
    print(len(predictions_kaggle))

    # format data for csv submission     
    c0 = []
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    c6 = []
    c7 = []
    c8 = []
    c9 = []
    for i in range(len(predictions_kaggle)):
        c0.append(predictions_kaggle[i][0])
        c1.append(predictions_kaggle[i][1])
        c2.append(predictions_kaggle[i][2])
        c3.append(predictions_kaggle[i][3])
        c4.append(predictions_kaggle[i][4])
        c5.append(predictions_kaggle[i][5])
        c6.append(predictions_kaggle[i][6])
        c7.append(predictions_kaggle[i][7])
        c8.append(predictions_kaggle[i][8])
        c9.append(predictions_kaggle[i][9])

    # convert data into a df in  img c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 format
    submission_dict = {'img':file_names,'c0':c0,'c1':c1,'c2':c2,'c3':c3,'c4':c4,'c5':c5,'c6':c6,'c7':c7,'c8':c8,'c9':c9}
    submission_df = pd.DataFrame(submission_dict)
    # re order the columns 
    submission_df = submission_df[[u'img',u'c0', u'c1', u'c2', u'c3', u'c4', u'c5', u'c6', u'c7', u'c8', u'c9']]
    # save as csv
    submission_df.to_csv(sub_file_name +'.csv',index = False)
    print("Subission file genrated with file name {}".format(sub_file_name))
    


# In[16]:



#Create submission file
create_submission_file(150,150,model,"VGG19_Kfold_submission")

