import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self) :
        self.params = {}
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)
        
    def init_params(self):
        # for tune
        np.random.seed(99)
        layer_input_ndim = 256
        layer_output_ndim = 10
        # initialize values of params(W & Bias) with glorot Xavier uniform
        limit = np.sqrt(6 / (layer_input_ndim + layer_output_ndim))
        self.params['W1'] = np.random.uniform(-limit, limit, size=(layer_input_ndim, layer_output_ndim)) # shapeL [n_in x n_out]
        # self.params['b1'] = np.zeros((1,layer_output_ndim)) # shape: [1 x n_out]

    def forward(self, input_X):
        W1 = self.params['W1']
        # b1 = self.params['b1']
        Z = np.dot(input_X,W1)
        self.logits = self.sigmoid(Z)
        return self.logits, Z
    
    def gradient_descent(self, train_X, diff, derivative, lr = 0.01):
        self.params['W1'] += lr * np.dot(train_X.T, np.multiply(diff, derivative))
        # self.params['b1'] += lr * diff.sum()
    
    def accuracy(self, train_X, train_Y):
        # foward pass and compare Y^ to Y_label to get acc
        logits, _= self.forward(train_X)
        y_pred = np.argmax(logits, axis=1)
        label = np.argmax(train_Y, axis=1)
        accuracy = (y_pred == label).mean()
        return accuracy
    
    def testing(self, test_X, test_Y):
        # get acc for test set
        accuracy = self.accuracy(test_X, test_Y)
        print()
        print('Accuracy of test: ', accuracy)
    
    def training(self, model, train_X, train_Y, epochs=1000, lr=0.01):
        # initialize params
        model.init_params()
        print('Training:')
        dict_loss_acc = dict()
        for index in range(epochs):
            logits, Z = model.forward(train_X)
            derivative = self.sigmoid_derivative(Z)
            diff = train_Y - logits
            loss = (diff**2).sum() / 2
            
            self.gradient_descent(train_X, diff, derivative, lr=lr)
            accuracy = self.accuracy(train_X, train_Y)
            dict_loss_acc[index] = [loss, accuracy]
            if index % 10 == 0:
                print('epoch: ', index, 'loss: ', loss, 'accuracy: ', accuracy)
        return dict_loss_acc

def plot_loss_acc(dict_loss_acc):
    epochs = [epoch for epoch in dict_loss_acc.keys()]
    loss = [val[0] for val in dict_loss_acc.values()]
    acc = [val[1] for val in dict_loss_acc.values()]
    
    # plot
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax1.plot(epochs, loss, color='orange')
    ax1.set_title('Loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')

    ax2 = fig.add_subplot(122)
    ax2.plot(epochs, acc, color='green')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy')
    plt.show()
    

    

        
            

if __name__ == "__main__":  

    # extract dataset
    train_set = pd.read_csv('./datasets/train_in.csv',header=None).values
    train_label = np.ravel(pd.read_csv('./datasets/train_out.csv',header=None).values)
    test_set = pd.read_csv('./datasets/test_in.csv',header=None).values
    test_label = np.ravel(pd.read_csv('./datasets/test_out.csv',header=None).values)

    # one hot encoding
    one_hot_label_train = np.zeros((len(train_label),10))
    one_hot_label_train[range(len(train_label)), train_label] = 1.0
    one_hot_label_test = np.zeros((len(test_label),10))
    one_hot_label_test[range(len(test_label)), test_label] = 1.0

    # insert bias
    bias_train = np.ones(train_set.shape[0])
    train_set = np.insert(train_set, 0, values=bias_train, axis=1)
    bias_test = np.ones(test_set.shape[0])
    test_set = np.insert(test_set, 0, values=bias_test, axis=1)

    # train & test
    model = Perceptron()
    dict_loss_acc = model.training(model, train_set, one_hot_label_train, epochs=4000, lr=0.0005)
    model.testing(test_set, one_hot_label_test)

    plot_loss_acc(dict_loss_acc)
