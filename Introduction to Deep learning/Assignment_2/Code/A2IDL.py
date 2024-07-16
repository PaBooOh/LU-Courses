#!/usr/bin/env python
# coding: utf-8

# # Task 1: Learn the basics of Keras and TensorFlow

# In[ ]:


# mnist_mlp.py (updated!)
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import rmsprop_v2
from keras import utils as np_utils


batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


# mnist_cnn.py (updated!)

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='Adadelta',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# - load MNIST and Fashion MNIST
# - a MLP described in detail in Ch. 10, pp. 297-307 and a CNN described in
# Ch. 14, p. 447.
# - trying various options: initializations, activations,
# training algorithms(optimizers) (and their hyperparameters), regularizations (L1, L2, Dropout, no Dropout).
# - changing the architecture of both networks: adding/removing layers,
# number of convolutional filters, their sizes

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import SGD
from functools import partial
from keras.callbacks import ModelCheckpoint, EarlyStopping


# In[ ]:


# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')
import datetime
# Clear any logs from previous runs
get_ipython().system('rm -rf ./logs/ ')


# In[ ]:


def mlp(dataset = 'mnist', numHiddenLayers = 1, numHiddenNeurons = 300, hiddenActivation = 'relu', 
        dropout = False, l1 = False, l2 = False, initializer = initializers.RandomNormal(stddev=0.05), 
        optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False), saysth = ''):

  mnist = getattr(keras.datasets, dataset)
  (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

  X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
  y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
  model = Sequential()
  model.add(Flatten(input_shape=[28, 28]))
  for layers in range(numHiddenLayers):
    if ((l1 == l2)&(l1==False)):
        model.add(Dense(numHiddenNeurons, activation=hiddenActivation, kernel_initializer=initializer))
        if (dropout == True):
          model.add(Dropout(0.2))
    if (l1 == True):
      model.add(Dense(numHiddenNeurons, activation=hiddenActivation, kernel_initializer=initializer, kernel_regularizer= keras.regularizers.l1_l2(l1=0.01, l2=0)))
    if (l2 == True):
      model.add(Dense(numHiddenNeurons, activation=hiddenActivation, kernel_initializer=initializer, kernel_regularizer= keras.regularizers.l1_l2(l1=0, l2=0.01)))

  model.add(Dense(10, activation="softmax", kernel_initializer=initializer))

  model.summary()

  model.compile(loss="sparse_categorical_crossentropy",
  optimizer=optimizer,
  metrics=["accuracy"])

  # specify tensorboard log dir
  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + saysth
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  # earlystop_callback = EarlyStopping()

  history = model.fit(X_train, y_train, epochs = 15,
                      validation_data=(X_valid, y_valid),
                      callbacks=[tensorboard_callback])


# In[ ]:


# vanilla mlp net
for i in range(3):
  mlp(saysth = 'vanilla')


# In[ ]:


# initialization: He Normal
for i in range(3):
  mlp(saysth = 'HeNormal', initializer = tf.keras.initializers.HeNormal())


# In[ ]:


# initialization: Xavier Normal
for i in range(3):
  mlp(saysth = 'GlorotNormal', initializer = tf.keras.initializers.HeNormal())


# In[ ]:


# initialization: Small Normal
for i in range(3):
  mlp(saysth = 'smallNormal', initializer = initializers.RandomNormal(stddev=0.001))


# In[ ]:


# initialization: Large Normal
for i in range(3):
  mlp(saysth = 'largeNormal', initializer = initializers.RandomNormal(stddev=10))


# In[ ]:


# tahn mlp net
for i in range(3):
    mlp(saysth='tanh', hiddenActivation = 'tanh')


# In[ ]:


# sigmoid mlp net
for i in range(3):
    mlp(saysth='sigmoid', hiddenActivation = 'sigmoid')


# In[ ]:


# optimizer hypermeters:
for i in range(3):
  mlp(saysth = 'momentum=0.9, nesterov=True', optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True))


# In[ ]:


mlp(saysth = 'momentum=0.9, nesterov=True', optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True))


# In[ ]:


# optimizer: RMSprop
for i in range(3):
    mlp(saysth= 'RMSprop', optimizer =tf.keras.optimizers.RMSprop())


# In[ ]:


# optimizer: Adam
for i in range(3):
    mlp(saysth= 'Adam', optimizer =tf.keras.optimizers.Adam())


# In[ ]:


# regulazitions: dropout
for i in range(3):
    mlp(dropout = True, saysth = 'dropoutTrue')


# In[ ]:


# regulazitions: L1
for i in range(3):
    mlp(l1 = True, saysth = 'L1')


# In[ ]:


# regulazitions: L2
for i in range(3):
    mlp(l2 = True, saysth = 'L2')


# In[ ]:


def cnn(dataset = 'mnist', hiddenActivation = 'relu', 
        dropout = True, l1 = False, l2 = False, initializer = initializers.RandomNormal(stddev=0.05), 
        optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False), saysth = ''):
    
    mnist = getattr(keras.datasets, dataset)
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    DefaultConv2D = partial(keras.layers.Conv2D,
      kernel_size=3, activation='relu', padding="SAME", kernel_initializer=initializer)
    model = Sequential()
    model.add(DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(DefaultConv2D(filters=128))
    model.add(DefaultConv2D(filters=128))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(DefaultConv2D(filters=256))
    model.add(DefaultConv2D(filters=256))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128, activation=hiddenActivation))
    if (dropout == True):
        model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=64, activation=hiddenActivation))
    if (dropout == True):
        model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=10, activation='softmax'))
    
    model.summary()

    model.compile(loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])
    
    # specify tensorboard log dir
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + saysth
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # earlystop_callback = EarlyStopping()

    history = model.fit(X_train, y_train, epochs = 30,
                      validation_data=(X_valid, y_valid),
                      callbacks=[tensorboard_callback])


# In[ ]:


# vallina CNN
cnn(saysth='vallina')


# In[ ]:


# initialization: He Normal
for i in range(3):
    cnn(saysth = 'HeNormal', initializer = tf.keras.initializers.HeNormal())


# In[ ]:


# optimizer hypermeters:
cnn(saysth = 'momentum=0.9, nesterov=True', optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True))
momentum=0.9, nesterov=True
cnn(saysth = 'momentum=0.9, nesterov=True', optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True))
# optimizer: RMSprop
cnn(saysth= 'RMSprop', optimizer =tf.keras.optimizers.RMSprop())
# optimizer: RMSprop
cnn(saysth= 'RMSprop', optimizer =tf.keras.optimizers.RMSprop())
# optimizer: Adam
cnn(saysth= 'Adam', optimizer =tf.keras.optimizers.Adam())


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')


# # Task 2: Testing the impact of obfuscating data by randomly permuting all pixels.

# In[ ]:


(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

for num in range(len(X_train_full)):
    X_train_full[num] = np.random.permutation(X_train_full[num].T).T
    X_train_full[num] = np.random.permutation(X_train_full[num])
for num in range(len(X_test)):
    X_test[num] = np.random.permutation(X_test[num].T).T
    X_test[num] = np.random.permutation(X_test[num])


# In[ ]:


import numpy as np

def mlp_permutation(dataset = 'mnist', numHiddenLayers = 1, numHiddenNeurons = 300, hiddenActivation = 'relu', 
        dropout = False, l1l2 = False, initializer = initializers.RandomNormal(stddev=0.05), 
        optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False), saysth = ''):

            
  X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
  y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
  model = Sequential()
  model.add(Flatten(input_shape=[28, 28]))
  for layers in range(numHiddenLayers):
    model.add(Dense(numHiddenNeurons, activation=hiddenActivation, kernel_initializer=initializer))
    if (dropout == True):
      model.add(Dropout(0.2))
    if (l1l2 == True):
      model.add(L1L2(l1=0.01, l2=0.01))

  model.add(Dense(10, activation="softmax", kernel_initializer=initializer))

  model.summary()

  model.compile(loss="sparse_categorical_crossentropy",
  optimizer=optimizer,
  metrics=["accuracy"])

  # specify tensorboard log dir
  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + saysth
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  # earlystop_callback = EarlyStopping()

  history = model.fit(X_train, y_train, epochs = 15,
                      validation_data=(X_valid, y_valid),
                      callbacks=[tensorboard_callback])


# In[ ]:


def cnn_permutation(dataset = 'mnist', hiddenActivation = 'relu', 
        dropout = True, l1 = False, l2 = False, initializer = initializers.RandomNormal(stddev=0.05), 
        optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False), saysth = ''):
    

    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    DefaultConv2D = partial(keras.layers.Conv2D,
      kernel_size=3, activation='relu', padding="SAME")
    model = Sequential()
    model.add(DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(DefaultConv2D(filters=128))
    model.add(DefaultConv2D(filters=128))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(DefaultConv2D(filters=256))
    model.add(DefaultConv2D(filters=256))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128, activation=hiddenActivation))
    if (dropout == True):
        model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=64, activation=hiddenActivation))
    if (dropout == True):
        model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=10, activation='softmax'))
    
    model.summary()

    model.compile(loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])
    
    # specify tensorboard log dir
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + saysth
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # earlystop_callback = EarlyStopping()

    history = model.fit(X_train, y_train, epochs = 15,
                      validation_data=(X_valid, y_valid),
                      callbacks=[tensorboard_callback])


# In[ ]:


mlp(saysth='mnist.mlp',dropout=True, initializer = tf.keras.initializers.HeNormal())


# In[ ]:


mlp_permutation(saysth='mnist.mlp.permutation',dropout=True, initializer = tf.keras.initializers.HeNormal())


# In[ ]:


cnn_permutation(saysth='mnist.cnn.permutation')

