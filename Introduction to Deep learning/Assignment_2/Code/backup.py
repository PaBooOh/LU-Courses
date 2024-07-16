import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Dense, Flatten, Activation, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn import model_selection
from tensorflow.python.keras.layers.core import Dropout

def split_dataset(test_size=0.2):
    # extract data
    labels = np.load('labels.npy')
    images = np.load('images.npy')
    # shuffle and split
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(images, labels, test_size = test_size, random_state = 55)
    Y_hours_train = Y_train[:,0]
    Y_minutes_train = Y_train[:,1]
    Y_hours_test = Y_test[:,0]
    Y_minutes_test = Y_test[:,1]
    # one-hot encode
    Y_hours_train = tf.one_hot(Y_hours_train, 12)
    Y_hours_test = tf.one_hot(Y_hours_test, 12)

    return X_train/255.0, X_test/255.0, Y_hours_train, Y_minutes_train, Y_hours_test, Y_minutes_test

def img_show(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

class Layer():    
    # multi-output (2 heads)
    def layer(self):
        # define shape (number of samples will be defined automatically)
        img_inputs = Input(shape=(150, 150, 1))
        # 1st
        X = Conv2D(32, 5, activation='relu')(img_inputs)
        # X = BatchNormalization()(X)
        X = MaxPooling2D((2, 2))(X)
        # 2rd
        X = Conv2D(32, 3, activation='relu')(X)
        # X = BatchNormalization()(X)
        X = MaxPooling2D((2, 2))(X)
        # 3th
        X = Conv2D(64, 3, activation='relu')(X)
        # X = BatchNormalization()(X)
        X = MaxPooling2D((2, 2))(X)
        # 4th
        X = Conv2D(64, 3, activation='relu')(X)
        # X = BatchNormalization()(X)
        X = MaxPooling2D((2, 2))(X)
        X = Dropout(0.25)(X) # dropout layer
        # fed into full connection layers
        X = Flatten()(X)

        # head-1
        hours = Dense(128, activation = 'relu')(X)
        hours = Dropout(0.5)(hours)
        hours = Dense(12, activation = 'softmax', name = 'hours')(hours)
        # head-2
        minutes = Dense(128, activation = 'relu')(X)
        # minutes = Dropout(0.1)(minutes)
        minutes = Dense(1, activation = 'linear', name = 'minutes')(minutes)

        model = Model(inputs = img_inputs, outputs = [hours, minutes])
        return model

class CNN():
    def __init__(self, dataset, lr=0.0001, epoch=10, batch_size=64):
        self.dataset = dataset
        # self.optim = SGD(lr=lr, momentum=0.6)
        self.optim = Adam(lr=lr)
        self.epoch = epoch
        self.batch_size = batch_size
    
    def training(self):
        # basic information
        loss = {'hours': 'categorical_crossentropy', 'minutes': 'mse'}
        acc = {'hours': 'acc', 'minutes': 'mae'}
        
        # start training
        model = Layer().layer()
        model.compile(optimizer=self.optim, loss=loss, metrics=acc)
        # dataset-> X_train, X_test, Y_hours_train, Y_minutes_train, Y_hours_test, Y_minutes_test
        model.fit(
            self.dataset[0], # X_train
            [self.dataset[2], self.dataset[3]], # Y_hours_train, Y_minutes_train
            validation_data = (self.dataset[1], [self.dataset[4], self.dataset[5]]),
            epochs = self.epoch,
            batch_size = self.batch_size
            )
        model.save('epoch200')
        # model.save_weights('epoch45.ckpt'.format(epoch=45))
    
    def testing(self, model_path, imgs):
        model = load_model(model_path)
        # latest = tf.train.latest_checkpoint('checkpoint_dir')
        # model = Layer().layer()
        # model.load_weights(latest)
        test_img = imgs.reshape((-1,150,150,1))
        pred = model.predict(test_img)
        hour = np.argmax(pred[0])
        minute = int(pred[1][0][0])
        print(str(hour),' ', str(minute))
        img_show(imgs)
    
    def evaluate(self, model_path, imgs, Y_1, Y_2):
        model = load_model(model_path)
        test_img = imgs.reshape((-1,150,150,1))
        acc = model.evaluate(test_img, [Y_1, Y_2], verbose=2)
        # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

        
if __name__ == "__main__":

    splited_dataset = split_dataset(test_size=0.2)
    # define hyper-parameters
    learning_rate = 0.001
    epoch = 400
    batch_size = 32
    cnn = CNN(splited_dataset, lr = learning_rate, epoch = epoch, batch_size = batch_size)
    cnn.training()
    # print(splited_dataset[2][12355], splited_dataset[3][12355])
    # cnn.testing('epoch20', splited_dataset[1][22])
    # cnn.evaluate('epoch20', splited_dataset[0], splited_dataset[2], splited_dataset[3])
   
    
