import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from sklearn import model_selection
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard

# process data into 720
def split_dataset(test_size=0.2):
    # extract data
    labels = np.load('labels.npy')
    images = np.load('images.npy')
    # transform labels from, say (0,32) to 0*60+32=32. Range: [0,719]
    Y = []
    for item in labels:
        computed = item[0] * 60 + item[1]
        Y.append(computed)
    
    # shuffle and split
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(images, Y, test_size = test_size, random_state = 55)

    # one-hot encode
    # Y_train = tf.one_hot(Y_train, 720)
    # Y_test = tf.one_hot(Y_test, 720)

    return X_train/255.0, X_test/255.0, np.array(Y_train), np.array(Y_test)
    # return X_train/255.0, X_test/255.0, Y_train, Y_test

def img_show(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


class Layer():    
    # 720 classes
    def layer(self):
        # define shape (number of samples will be defined automatically)
        img_inputs = Input(shape=(150, 150, 1))
        # 1st
        X = Conv2D(32, 3, activation='relu')(img_inputs)
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
        X = Dropout(0.25)(X)
        X = Flatten()(X)

        X = Dense(128, activation = 'relu')(X)
        X = Dropout(0.25)(X)
        X = Dense(720, activation = 'softmax')(X)

        model = Model(inputs = img_inputs, outputs = X)
        return model

class CNN():
    def __init__(self, dataset, lr=0.0001, epoch=10, batch_size=64):
        self.dataset = dataset
        # self.optim = SGD(learning_rate=lr, decay=1e-6, momentum=0.9)
        # self.optim = RMSprop(learning_rate=lr, decay=1e-6)
        self.optim = Adam(learning_rate=lr)
        self.epoch = epoch
        self.batch_size = batch_size
    
    def training(self, tensorboard_dir=None):   
        # start training
        model = Layer().layer()
        print(model.summary())
        model.compile(optimizer=self.optim, loss='sparse_categorical_crossentropy', metrics='acc')
        log_dir = tensorboard_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.fit(
            self.dataset[0], # X_train
            self.dataset[2], # Y_train
            validation_data = (self.dataset[1], self.dataset[3]), # X_test, Y_test
            epochs = self.epoch,
            batch_size = self.batch_size,
            callbacks = [tensorboard_callback]
            )
        model.save('cls_epoch400_1')
        # model.save_weights('epoch45.ckpt'.format(epoch=45))
    
    def testing(self, model_path, imgs):
        model = load_model(model_path)
        test_img = imgs.reshape((-1,150,150,1))
        pred = model.predict(test_img)
        hour = np.argmax(pred[0])
        minute = int(pred[1][0][0])
        print(str(hour),' ', str(minute))
        img_show(imgs)
    
    def evaluate(self, model_path, imgs, labels):
        model = load_model(model_path)
        # print(model.summary())
        test_img = imgs.reshape((-1,150,150,1))
        model.evaluate(test_img, labels, verbose=2)
    
    def retrain(self, model_path, epochs, batch_size, lr):
        model = load_model(model_path)
        model.compile(optimizer = Adam(learning_rate=lr), loss = 'categorical_crossentropy', metrics = 'acc')
        log_dir = "logs/cls_fit_1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        model.fit(
            self.dataset[0], # X_train
            self.dataset[2], # Y_train
            validation_data = (self.dataset[1], self.dataset[3]), # X_test, Y_test
            epochs = epochs,
            batch_size = batch_size,
            callbacks = [tensorboard_callback]
            )
        model.save('cls_epoch400_1')

if __name__ == "__main__":

    splited_dataset = split_dataset(test_size=0.2)
    # define hyper-parameters
    learning_rate = 0.0001
    epoch = 200
    batch_size = 32
    # print(splited_dataset[2][0])
    cnn = CNN(splited_dataset, lr = learning_rate, epoch = epoch, batch_size = batch_size)
    cnn.training(tensorboard_dir='logs/cls_fit_2/')
    # cnn.testing('cls_epoch200_', splited_dataset[0][12355])
    # cnn.evaluate('cls_epoch400_new', splited_dataset[1], splited_dataset[3])
    # cnn.retrain('cls_epoch200_', 400, batch_size, 0.0001)
