#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils


# In[5]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[6]:


# Reshaping to format which CNN expects (batch, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')


# In[7]:


# normalize inputs from 0-255 to 0-1
X_train/=255
X_test/=255


# In[8]:


# one hot encode
number_of_classes = 10
y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)


# In[9]:


# create model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(number_of_classes, activation='softmax'))


# In[10]:


# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


# In[11]:


# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)


# In[12]:


# Save the model
model.save('model.h5')


# In[13]:


# Final evaluation of the model
metrics = model.evaluate(X_test, y_test, verbose=0)
print("Metrics(Test loss & Test Accuracy): ")
print(metrics)


# In[14]:


pred = model.predict(X_train)
print(pred)


# In[15]:


import numpy as np
predict_classes = np.argmax(pred,axis=1)
predict_classes


# In[16]:


expected_classes = np.argmax(y_train,axis=1)
expected_classes


# In[17]:


from sklearn.metrics import accuracy_score
correct = accuracy_score(expected_classes,predict_classes)
correct


# In[20]:


import os
path='./'
model_json = model.to_json()
with open(os.path.join(path,"model.json"), "w") as json_file:
    json_file.write(model_json)


# In[18]:




