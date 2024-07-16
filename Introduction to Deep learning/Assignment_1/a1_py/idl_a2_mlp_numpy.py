import numpy as np
import pandas as pd

class CrossEntropyLoss():
    def __init__(self, batch_logits, batch_label) :
        self.batch_logits = batch_logits
        self.batch_label = batch_label

    def softmax(self, batch_logits):
        exps = np.exp(batch_logits)
        return exps/np.sum(exps,axis=-1,keepdims=True)
    
    def derivative(self):
        return self.batch_logits - self.batch_label
    
    def loss(self):
        props = self.batch_logits # already activated by softmax outside
        # divide by samples_n, get loss(scalar)
        loss = -np.einsum('ij,ij->', self.batch_label ,np.log(props), optimize=True) / self.batch_label.shape[0]
        return loss


class MLP:
    def __init__(self, network) :
        self.network = network
        self.params = {}

    # define activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)
    def relu(self, x):
        return np.max(0, x)
    def relu_derivative(self, x):
        return np.greater(x, 0).astype(int)
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    def tanh_derivative(self, x):
        tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return 1 - tanh ** 2
    def softmax(self, x):
        exps = np.exp(x-np.max(x,axis=-1, keepdims=True))
        return exps/np.sum(exps,axis=-1,keepdims=True)

    # initialize W and bias
    def init_params(self):
        np.random.seed(123)
        for index, layer in enumerate(self.network):
            # initialize strucure of params matrices
            layer_input_ndim = layer['in_features']
            layer_output_ndim = layer['out_features']
            # initialize values of params(W & Bias) with glorot Xavier uniform
            limit = np.sqrt(6 / (layer_input_ndim + layer_output_ndim))
            self.params['W' + str(index + 1)] = np.random.uniform(-limit,limit,size=(layer_input_ndim,layer_output_ndim)) # shapeL [n_in x n_out]
            print(layer_input_ndim, layer_output_ndim)
            self.params['b' + str(index + 1)] = np.zeros((1,layer_output_ndim)) # shape: [1 x n_out]
    
    # forward pass
    def forward(self, input):
        bp = {} # stored for backward pass
        A = self.input = input # A represents output in layers(after activating), input.shape [batch_size x n_features]
        for index, layer in enumerate(self.network):
            ac_func = layer['activ']
            W = self.params['W' + str(index + 1)]
            b = self.params['b' + str(index + 1)]
            Z = np.dot(A,W) + b # input in layers
            
            # activation function
            # if ac_func == 'relu':
            #     A = self.relu(Z)
            if ac_func == 'sigmoid':
                A = self.sigmoid(Z)
            elif ac_func == 'tanh':
                A = self.tanh(Z)
            elif ac_func == 'softmax' and index == len(self.network) - 1:
                A = self.softmax(Z) # activated by softmax(last layer)
            else:
                raise Exception('No such activation function or softmax used wrongly.')
            bp['A' + str(index + 1)] = A
            bp['Z' + str(index + 1)] = Z
            bp['W' + str(index + 1)] = W
            bp['b' + str(index + 1)] = b
        # print(np.sum(A[2]))
        self.bp_stored = bp
        return A, bp
    
    # backward pass
    def backward(self,batch_label, bp_stored, params):
        '''
        # rough chain rule w.r.t softmax_cross_entropy
        # last layer only
        dz2 = a2-y
        dw2 = dz2*a1
        db2 = dz2
        # following applied for all layers except for the last layer
        dz1 = w2*dz2*activ_1'(z1)
        dw1 = dz1*a0
        db1 = dz1
        #
        dL/dw1 = dz2*w2*g
        '''
        layers_num = len(self.network)
        grads = {}
        samples_num = batch_label.shape[0] # batch_size
        bp_stored['A0'] = self.input # shape (batch_size, 256)
        # (1) exclusive computation for last layer
        A_logits = bp_stored['A' + str(layers_num)] # shape: (batch_size, 10)
        A_prev = bp_stored['A' + str(layers_num - 1)] # shape: (batch_size, 128)
        # loss_func = CrossEntropyLoss(A_logits,batch_label)
        # dZ_final = loss_func.derivative() # dL/dzi
        dZ_final = A_logits - batch_label # dL/dzi
        dW_final = np.dot(A_prev.T, dZ_final) / samples_num # shape: (batch_size, 128).T * (batch_size, 10) ==> (128, 10)
        db_final = np.sum(dZ_final, axis=0, keepdims=True) / samples_num
        grads['dW' + str(layers_num)] = dW_final
        grads['db' + str(layers_num)] = db_final
        dA_prev = np.dot(dZ_final, dW_final.T) # dL/dW1 = dL/dz2 * dz2/da1 * .. , where dL/da1 = dz2 * (d(a1*w2+b2)/da1) = dz2 * w2

        # (2) scenario in other layers except for the last layer
        for layer_index in range(layers_num - 1, 0, -1):
            # print(bp_stored['Z' + str(layer_index)].shape)
            if self.network[layer_index-1]['activ'] == 'sigmoid':
                dZ_prev = dA_prev * self.sigmoid_derivative(bp_stored['Z' + str(layer_index)]) # dL/da1 = da1 = dz2 * w2, dL/dz1 = dz1 = da1 * activ'(z1)
            elif self.network[layer_index-1]['activ'] == 'relu':
                dZ_prev = dA_prev * self.relu_derivative(bp_stored['Z' + str(layer_index)]) # dL/da1 = da1 = dz2 * w2, dL/dz1 = dz1 = da1 * activ'(z1)
            elif self.network[layer_index-1]['activ'] == 'tanh':
                dZ_prev = dA_prev * self.tanh_derivative(bp_stored['Z' + str(layer_index)]) # dL/da1 = da1 = dz2 * w2, dL/dz1 = dz1 = da1 * activ'(z1)
            dW_prev = 1. / samples_num * np.dot(bp_stored['A' + str(layer_index - 1)].T, dZ_prev)  # shape (256, 128)
            db_prev = 1. / samples_num * np.sum(dZ_prev, axis=0, keepdims=True)
            if layer_index > 1:
                dA_prev = np.dot(dZ_prev, dW_prev.T)
            grads['dW' + str(layer_index)] = dW_prev
            grads['db' + str(layer_index)] = db_prev
        self.grads = grads
        return grads
    
    # upadate params
    def gradient_descent(self, learning_rate = 0.1):
        for layer_index in range(1, len(self.network)+1):
            # print(self.params['b' + str(layer_index)].shape)
            self.params['W' + str(layer_index)] -= learning_rate * self.grads['dW' + str(layer_index)]        
            self.params['b' + str(layer_index)] -= learning_rate * self.grads['db' + str(layer_index)]
        return self.params
            
    def accuracy(self, train_X, train_Y):
        # foward pass and compare Y^ to Y_label to get acc
        A_logits, _ = self.forward(train_X)
        y_pred = np.argmax(A_logits, axis=1)
        label = np.argmax(train_Y, axis=1)
        accuracy = (y_pred == label).mean()
        return accuracy

    def testing(self, model, test_X, test_Y):

        # one-hot encoding for label_Y
        one_hot_label = np.zeros((len(test_Y),10))
        one_hot_label[range(len(test_Y)), test_Y] = 1.0
        
        accuracy = self.accuracy(test_X, one_hot_label)
        print('test_accuracy: ', accuracy)



    def training(self, model, train_X, train_Y, epochs=1000, batch_size=500, lr=0.1):

        # one-hot encoding for label_Y
        one_hot_label = np.zeros((len(train_Y),10))
        one_hot_label[range(len(train_Y)), train_Y] = 1.0
        samples_num = len(train_Y)

        # initialize params
        
        params = model.init_params()
        # training
        for _ in range(epochs):
            lst_labels = np.array_split(one_hot_label, samples_num//batch_size + 1)
            lst_trainX = np.array_split(train_X, samples_num//batch_size + 1)

            # for i in range(len(lst_trainX)):
            #     print(len(lst_trainX[i]))
            for iter in range(samples_num//batch_size):
                # forward pass
                batch_logits, bp_stored = model.forward(lst_trainX[iter])
                # calculate loss
                loss_func = CrossEntropyLoss(batch_logits, lst_labels[iter])
                loss = loss_func.loss()
                accuracy = self.accuracy(lst_trainX[iter], lst_labels[iter])
                # backward pass
                model.backward(lst_labels[iter], bp_stored, params)
                # update params
                params = model.gradient_descent(learning_rate=lr)
                print('loss: ', loss, 'accuracy: ', accuracy)
                

if __name__ == "__main__":  
    # extract dataset
    train_set = pd.read_csv('./datasets/train_in.csv',header=None).values
    train_label = np.ravel(pd.read_csv('./datasets/train_out.csv',header=None).values)

    test_set = pd.read_csv('./datasets/test_in.csv',header=None).values
    test_label = np.ravel(pd.read_csv('./datasets/test_out.csv',header=None).values)
    # decide on strcture of nn
    strcture = [
        {'in_features':16 * 16, 'out_features': 1024, 'activ': 'tanh'}, # fc1: hidden layer
        {'in_features':1024, 'out_features': 10, 'activ': 'softmax'} # fc2: output layer
    ]
    # model
    mlp = MLP(network=strcture)
    mlp.training(mlp, train_set, train_label)
    mlp.testing(mlp, test_set, test_label)

