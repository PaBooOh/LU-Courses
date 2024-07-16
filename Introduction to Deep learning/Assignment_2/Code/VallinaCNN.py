import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import SGD
from functools import partial

# # Load the TensorBoard notebook extension
# %load_ext tensorboard
# import datetime
# # Clear any logs from previous runs
# !rm -rf ./logs/ 


def cnn(dataset = 'mnist', hiddenActivation = 'relu', 
        dropout = True, l1 = False, l2 = False, initializer = initializers.RandomNormal(stddev=0.05), 
        optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False), saysth = ''):
    
    mnist = getattr(keras.datasets, dataset)
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

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
    log_dir = "logs/fitt/" + saysth
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # earlystop_callback = EarlyStopping()

    history = model.fit(X_train, y_train, epochs = 15,
                      validation_data=(X_valid, y_valid),
                      callbacks=[tensorboard_callback])

cnn(saysth='vallina')