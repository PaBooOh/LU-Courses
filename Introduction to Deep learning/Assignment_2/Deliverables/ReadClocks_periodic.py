import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import tensorflow.python.ops.numpy_ops.np_config as np_config
from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D, Dense, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from sklearn import model_selection

def mse_sine(y_true, y_pred):
    np_config.enable_numpy_behavior()
    # rad = np.deg2rad((y_true-y_pred)/4)
    y_true = tf.cast(y_true, y_pred.dtype)
    deg = (y_true - y_pred) / 4.0
    rad = deg2rad(deg)
    # rad = tf.experimental.numpy.deg2rad(deg)
    error = tf.sin(rad)
    sqr_error = K.square(error)
    mse = K.sum(sqr_error)
    return mse

def deg2rad(deg):
    pi_factor = 0.017453292519943295
    return deg * pi_factor

def split_dataset(test_size=0.2):
    # extract data
    labels = np.load('labels.npy')
    images = np.load('images.npy')
    # transform labels from, say (0,32) to 0*60+32=32. Range: [0,719]
    Y = []
    for item in labels:
        computed = item[0] * 60 + item[1]
        coord = np.array([math.sin(2 * math.pi * computed / 720), math.cos(2 * math.pi * computed / 720), computed])
        Y.append(coord)
    
    # shuffle and split
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(images, Y, test_size = test_size, random_state = 55)
    Y_cosx_train = np.array(Y_train)[:,1]
    Y_siny_train = np.array(Y_train)[:,0]
    Y_cosx_test = np.array(Y_test)[:,1]
    Y_siny_test = np.array(Y_test)[:,0]
    Y_minutes_train = np.array(Y_train)[:,2]
    Y_minutes_test = np.array(Y_test)[:,2]

    return X_train/255.0, X_test/255.0, Y_siny_train, Y_cosx_train, Y_siny_test, Y_cosx_test, Y_minutes_train, Y_minutes_test

def split_dataset_4coords(test_size=0.2):
    # extract data
    labels = np.load('labels.npy')
    images = np.load('images.npy')
    # transform labels from, say (0,32) to 0*60+32=32. Range: [0,719]
    Y = []
    for item in labels:
        hours_coord = [math.sin(2 * math.pi * item[0] / 12), math.cos(2 * math.pi * item[0] / 12), item[0]]
        minutes_coord = [math.sin(2 * math.pi * item[1] / 60), math.cos(2 * math.pi * item[1] / 60), item[1]]
        Y.append([hours_coord, minutes_coord])
    
    # shuffle and split
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(images, Y, test_size = test_size, random_state = 55)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    Y_hours_siny_train = Y_train[:,0,0]
    Y_hours_cosx_train = Y_train[:,0,1]
    Y_hours_train = Y_train[:,0,2]
    Y_minutes_siny_train = Y_train[:,1,0]
    Y_minutes_cosx_train = Y_train[:,1,1]
    Y_minutes_train = Y_train[:,1,2]
    
    Y_train = np.array([Y_hours_siny_train, Y_hours_cosx_train, Y_minutes_siny_train, Y_minutes_cosx_train, Y_hours_train, Y_minutes_train])

    Y_hours_siny_test = Y_test[:,0,0]
    Y_hours_cosx_test = Y_test[:,0,1]
    Y_hours_test = Y_test[:,0,2]

    Y_minutes_siny_test = Y_test[:,1,0]
    Y_minutes_cosx_test = Y_test[:,1,1]
    Y_minutes_test = Y_test[:,1,2]
    Y_test = np.array([Y_hours_siny_test, Y_hours_cosx_test, Y_minutes_siny_test, Y_minutes_cosx_test, Y_hours_test ,Y_minutes_test])

    return X_train/255.0, X_test/255.0, Y_train, Y_test

def img_show(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

def mse_sine(y_true, y_pred):
    np_config.enable_numpy_behavior()
    deg = (y_true - y_pred) / 4
    rad = tf.experimental.numpy.deg2rad(deg)
    error = tf.sin(rad)
    sqr_error = K.square(error)
    mse = K.mean(sqr_error)
    return mse

def mse_radius(y_true, y_pred):
    deg = y_true - y_pred
    rad = tf.experimental.numpy.deg2rad(deg)
    error = tf.sin(rad)
    sqr_error = K.square(error)
    mse = K.mean(sqr_error)
    return mse

class Layer():    
    # multi-output (2 heads)
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

        # head-1
        siny = Dense(128, activation = 'relu')(X)
        siny = Dropout(0.25)(siny)
        siny = Dense(1, activation = 'linear', name = 'siny')(siny)
        # head-2
        cosx = Dense(128, activation = 'relu')(X)
        cosx = Dropout(0.25)(cosx)
        cosx = Dense(1, activation = 'linear', name = 'cosx')(cosx)

        model = Model(inputs = img_inputs, outputs = [siny, cosx])
        return model

class Layer_4heads():    
    # multi-output (2 heads)
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
        # head-1
        hours_siny = Dense(128, activation = 'relu')(X)
        hours_siny = Dropout(0.25)(hours_siny)
        hours_siny = Dense(1, activation = 'linear', name = 'hours_siny')(hours_siny)
        # head-2
        hours_cosx = Dense(128, activation = 'relu')(X)
        hours_cosx = Dropout(0.25)(hours_cosx)
        hours_cosx = Dense(1, activation = 'linear', name = 'hours_cosx')(hours_cosx)
        # head-3
        minutes_siny = Dense(128, activation = 'relu')(X)
        minutes_siny = Dropout(0.25)(minutes_siny)
        minutes_siny = Dense(1, activation = 'linear', name = 'minutes_siny')(minutes_siny)
        # head-4
        minutes_cosx = Dense(128, activation = 'relu')(X)
        minutes_cosx = Dropout(0.25)(minutes_cosx)
        minutes_cosx = Dense(1, activation = 'linear', name = 'minutes_cosx')(minutes_cosx)

        # model = Model(inputs = img_inputs, outputs = X)
        model = Model(inputs = img_inputs, outputs = [hours_siny, hours_cosx, minutes_siny, minutes_cosx,])
        return model

class CNN():
    def __init__(self, dataset, lr=0.0001, epoch=10, batch_size=64):
        self.dataset = dataset
        # self.optim = SGD(lr=lr, momentum=0.6)
        self.optim = Adam(learning_rate=lr)
        self.epoch = epoch
        self.batch_size = batch_size
    
    def training(self, tensorboard_dir=None):
        loss = {'siny': 'mse', 'cosx': 'mse'}
        acc = {'siny': 'mae', 'cosx': 'mae', }  
        # start training
        model = Layer().layer()
        # model.summary()
        model.compile(optimizer=self.optim, loss=loss, metrics=acc)
        log_dir = tensorboard_dir
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        model.fit(
            self.dataset[0], # X_train
            [self.dataset[2], self.dataset[3]], # Y_siny_train, Y_cosx_train
            validation_data = (self.dataset[1], [self.dataset[4], self.dataset[5]]),
            epochs = self.epoch,
            batch_size = self.batch_size,
            callbacks = [tensorboard_callback]
            )
        model.save('periodic_epoch200_3')
    
    def training_4coords(self, tensorboard_dir=None):
        loss = {'hours_cosx': 'mse', 'hours_siny': 'mse', 'minutes_cosx': 'mse', 'minutes_siny': 'mse'}
        acc = {'hours_cosx': 'mae', 'hours_siny': 'mae', 'minutes_cosx': 'mae', 'minutes_siny': 'mae'}  
        # start training
        model = Layer_4heads().layer()
        # model.summary()
        model.compile(optimizer=self.optim, loss=loss, metrics=acc)
        log_dir = tensorboard_dir
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        model.fit(
            self.dataset[0], # X_train
            [self.dataset[2][0], self.dataset[2][1], self.dataset[2][2], self.dataset[2][3]],
            validation_data = (self.dataset[1], [self.dataset[3][0], self.dataset[3][1], self.dataset[3][2], self.dataset[3][3]]),
            epochs = self.epoch,
            batch_size = self.batch_size,
            callbacks = [tensorboard_callback]
            )
        model.save('periodic_epoch1500_tst1')
    
    def testing(self, model_path, imgs):
        model = load_model(model_path)
        test_img = imgs.reshape((-1,150,150,1))
        pred = model.predict(test_img)
        # cosx = pred[0][0]
        # siny = pred[0][1]
        siny = pred[0][0]
        cosx = pred[1][0]
        
        theta = np.rad2deg(math.atan2(cosx, siny))
        print(theta)
        theta = (theta + 360) * 2 if theta < 0 else theta
        hours = int(theta / 60)
        minutes = int(theta % 60)
        print(str(hours), str(minutes))
        img_show(imgs)
    
    def testing_4coords(self, model_path, imgs):
        model = load_model(model_path)
        test_img = imgs.reshape((-1,150,150,1))
        pred = model.predict(test_img)
        hours_cosx = pred[1]
        hours_siny = pred[0]
        minutes_cosx = pred[3]
        minutes_siny = pred[2]
        deg_h = np.rad2deg(math.atan2(hours_cosx, hours_siny))
        deg_m = np.rad2deg(math.atan2(minutes_cosx, minutes_siny))
        theta_h = (360 + deg_h) if deg_h < 0 else deg_h
        theta_m = (360 + deg_m) if deg_m < 0 else deg_m
        
        # print(theta_h, theta_m)
        hours = round(theta_h / 360 * 12)
        minutes = round(theta_m / 360 * 60)
        print(str(hours), str(minutes))
        img_show(imgs)
    
    def evaluate(self, model_path, imgs, Y):
        mae_sum = 0
        model = load_model(model_path)
        for index, img in enumerate(imgs):
            test_img = img.reshape((-1,150,150,1))
            pred = model.predict(test_img)
            siny = pred[0][0]
            cosx = pred[1][0]
            
            
            theta = np.rad2deg(math.atan2(cosx, siny))
            minutes = (theta + 360) * 2 if theta < 0 else theta
            mae_sum = mae_sum + np.abs(minutes - Y[index])
        
        print(mae_sum / Y.shape[0])
    
    def evaluate_4coords(self, model_path, imgs, Y_hours, Y_minutes):
        mae_sum = 0
        model = load_model(model_path)
        for index, img in enumerate(imgs):
            test_img = img.reshape((-1,150,150,1))
            pred = model.predict(test_img)
            hours_siny = pred[0]
            hours_cosx = pred[1]
            minutes_siny = pred[2]
            minutes_cosx = pred[3]
            deg_h = np.rad2deg(math.atan2(hours_cosx, hours_siny))
            deg_m = np.rad2deg(math.atan2(minutes_cosx, minutes_siny))
            theta_h = (360 + deg_h) if deg_h < 0 else deg_h
            theta_m = (360 + deg_m) if deg_m < 0 else deg_m
            hours = round(theta_h / 360 * 12)
            if hours == 12:
                hours = 0 
            minutes = round(theta_m / 360 * 60)

            total_minutes = hours * 60 + minutes
            Y_total_minutes = Y_hours[index]*60 + Y_minutes[index]
            error = np.abs(total_minutes - Y_total_minutes)
            mae_sum = mae_sum + error
            # if error > 0:
            #     print(index, Y_hours[index], Y_minutes[index])
            #     print(str(hours), str(minutes))
        print(mae_sum / 3600)
    
    def retrain_4coords(self, model_path, epochs, batch_size, lr, tensorboard_dir=None):
        loss = {'hours_cosx': 'mse', 'hours_siny': 'mse', 'minutes_cosx': 'mse', 'minutes_siny': 'mse'}
        acc = {'hours_cosx': 'mae', 'hours_siny': 'mae', 'minutes_cosx': 'mae', 'minutes_siny': 'mae'}
        log_dir = tensorboard_dir
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        model = load_model(model_path)
        model.compile(optimizer = Adam(learning_rate=lr), loss = loss, metrics = acc)
        model.fit(
            self.dataset[0], # X_train
            [self.dataset[2][0], self.dataset[2][1], self.dataset[2][2], self.dataset[2][3]], # Y_cosx_train, Y_siny_train
            validation_data = (self.dataset[1], [self.dataset[3][0], self.dataset[3][1], self.dataset[3][2], self.dataset[3][3]]),
            epochs = self.epoch,
            batch_size = self.batch_size,
            callbacks = [tensorboard_callback]
            )
        model.save('periodic_100')
    
    def retrain(self, model_path, epochs, batch_size, lr):
        loss = {'hours': 'mse', 'minutes': 'mse'}
        acc = {'hours': 'mae', 'minutes': 'mae'}
        model = load_model(model_path)
        model.compile(optimizer = Adam(learning_rate=lr), loss = loss, metrics = acc)
        model.fit(
            self.dataset[0], # X_train
            [self.dataset[2], self.dataset[3]], # Y_hours_train, Y_minutes_train
            validation_data = (self.dataset[1], [self.dataset[4], self.dataset[5]]),
            epochs = epochs,
            batch_size = batch_size
            )
        model.save('epoch400_new')

        
if __name__ == "__main__":

    # define hyperparameters
    learning_rate = 0.0001
    epoch = 300
    batch_size = 64

    # ———————————————————————————— training and analysis——————————————————————————————
    '''
    1) optimal try(method)
    split labels consisting hours and minutes into two coordinate points respectively
    '''
    # splited_dataset = split_dataset_4coords(test_size=0.2)
    # cnn = CNN(splited_dataset, lr = learning_rate, epoch = epoch, batch_size = batch_size)
    # cnn.training_4coords(tensorboard_dir='logs/periodic_fit_tst1')
    # cnn.testing_4coords('periodic_epoch200_3', splited_dataset[1][967])
    # cnn.evaluate_4coords('periodic_epoch200_3', splited_dataset[1], splited_dataset[3][4], splited_dataset[3][5])
    # cnn.retrain_4coords('periodic_epoch200_3', 400, batch_size, 0.0001)

    '''
    2) a try close to the optimal one
    change labels to 720 categories and split them into a coordinate point (exist diff of 40 minutes)
    '''
    splited_dataset = split_dataset(test_size=0.2)
    cnn = CNN(splited_dataset, lr = learning_rate, epoch = epoch, batch_size = batch_size)
    # cnn.training(tensorboard_dir='logs/periodic_fit_2')
    cnn.testing('periodic_epoch200_0', splited_dataset[1][0])
    # cnn.evaluate('periodic_epoch200_0', splited_dataset[1], splited_dataset[7])
    # cnn.retrain('periodic_epoch600_0', 400, batch_size, 0.0001)
    
   
    
