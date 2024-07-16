import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import math
import tensorflow.python.ops.numpy_ops.np_config as np_config
from sklearn import model_selection
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from sympy import sin, cos, pi, atan2

# process data into 720
def split_dataset(test_size=0.2):
    # extract data
    labels = np.load('labels.npy')
    images = np.load('images.npy')
    # transform label from, say (0,32) to 0*60+32=32. Range: [0,719]
    Y = []
    for item in labels:
        computed = item[0] + round(item[1]/60, 2)
        Y.append(computed)
    # shuffle and split
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(images, Y, test_size = test_size, random_state = 55)

    return X_train/255.0, X_test/255.0, np.array(Y_train), np.array(Y_test)

def img_show(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

class Layer():    
    # 720 classes
    def layer(self):
        # define shape (number of samples will be defined automatically)
        img_inputs = Input(shape=(150, 150, 1))
        # 1st
        X = Conv2D(32, 5, activation='relu')(img_inputs)
        X = MaxPooling2D((2, 2))(X)
        # 2rd
        X = Conv2D(64, 3, activation='relu')(X)
        X = MaxPooling2D((2, 2))(X)
        # 3th
        X = Conv2D(128, 3, activation='relu')(X)
        X = MaxPooling2D((2, 2))(X)
        # 4th
        X = Conv2D(256, 3, activation='relu')(X)
        X = MaxPooling2D((2, 2))(X)
        X = Dropout(0.25)(X) # dropout layer
        # fed into full connection layers
        X = Flatten()(X)

        X = Dense(128, activation = 'relu')(X)
        X = Dropout(0.5)(X)
        X = Dense(1, activation = 'linear')(X)

        model = Model(inputs = img_inputs, outputs = X)
        return model

class CNN():
    def __init__(self, dataset, lr=0.0001, epoch=10, batch_size=64):
        self.dataset = dataset
        # self.optim = SGD(lr=lr, momentum=0.6)
        self.optim = Adam(learning_rate=lr)
        self.epoch = epoch
        self.batch_size = batch_size
    
    def training(self, tensorboard_dir=None):   
        # start training
        model = Layer().layer()
        model.compile(optimizer=self.optim, loss='mse', metrics='mae')
        log_dir = tensorboard_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        model.fit(
            self.dataset[0],
            self.dataset[2],
            validation_data = (self.dataset[1], self.dataset[3]),
            epochs = self.epoch,
            batch_size = self.batch_size,
            callbacks = [tensorboard_callback]
            )
        model.save('reg_epoch1200_1')
        # model.save_weights('epoch45.ckpt'.format(epoch=45))
    
    def testing(self, model_path, imgs):
        model = load_model(model_path)
        test_img = imgs.reshape((-1,150,150,1))
        pred = model.predict(test_img)
        hours = int(pred[0][0])
        minutes = round((pred[0][0] - hours) * 60)
        print(str(hours), str(minutes))
        img_show(imgs)
    
    def evaluate(self, model_path, imgs, labels):
        model = load_model(model_path)
        test_img = imgs.reshape((-1,150,150,1))
        model.evaluate(test_img, labels, verbose=2)
    
    def retrain(self, model_path, epochs, batch_size, lr, save_path):
        model = load_model(model_path)
        model.compile(optimizer = Adam(learning_rate=lr), loss = 'mse', metrics = 'mae')
        log_dir = "logs/reg_fit_5/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        model.fit(
            self.dataset[0], # X_train
            self.dataset[2], # Y_train
            validation_data = (self.dataset[1], self.dataset[3]), # X_test, Y_test
            epochs = epochs,
            batch_size = batch_size,
            callbacks = [tensorboard_callback]
            )
        model.save(save_path)

if __name__ == "__main__":
    # define hyper-parameters
    learning_rate = 0.0001
    epoch = 200
    batch_size = 32

    splited_dataset = split_dataset(test_size=0.2)
    cnn = CNN(splited_dataset, lr = learning_rate, epoch = epoch, batch_size = batch_size)
    cnn.training(tensorboard_dir='logs/reg_fit_2/')
    # cnn.evaluate('', splited_dataset[0])
    # cnn.testing('reg_epoch600_1', splited_dataset[1][59])
    # cnn.retrain('reg_epoch600_0', 600, batch_size, 0.0001, 'reg_epoch_600_1')