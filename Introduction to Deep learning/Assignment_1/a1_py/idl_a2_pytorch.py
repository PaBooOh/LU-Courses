
import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# extract dataset
train_set = pd.read_csv('./datasets/train_in.csv',header=None).values
train_label = pd.read_csv('./datasets/train_out.csv',header=None).values
test_set = pd.read_csv('./datasets/test_in.csv',header=None).values
test_label = pd.read_csv('./datasets/test_out.csv',header=None).values

# define sequence
class MulticlassPerceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.hidden_layer0 = nn.Linear(input_size, 256)
        self.output_layer = nn.Linear(256, output_size)
        self.droput = nn.Dropout(0.2)
    
    def forward(self,X):
        X = torch.sigmoid(self.hidden_layer0(X))
        X = self.output_layer(X)
        return X

# rewrite to obtain custom dataloader
class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, set_X, set_Y):
        self.data_X = set_X
        self.label_Y = set_Y
    def __getitem__(self, index):
        data = self.data_X[index]
        label = self.label_Y[index]
        return data, label
    def __len__(self):
        return len(self.data_X)



# define hyperparameters
lr = 0.001 # learning rate
epoch = 50
batch_size = 3

# get dataloader for training model
torch_dataset_train = MnistDataset(train_set, np.ravel(train_label))
torch_dataset_test = MnistDataset(test_set, np.ravel(test_label))
train_loader = Data.DataLoader(torch_dataset_train, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(torch_dataset_test, batch_size=len(test_label), shuffle=False)

mlp_model = MulticlassPerceptron(input_size = 256, output_size = 10)
optim = torch.optim.SGD(mlp_model.parameters(), lr=lr, momentum=0.9)
loss_function = nn.CrossEntropyLoss()
# print(mlp)

# training
def training(train_loader, model, optim, loss_func): 
    for i in range(epoch):
        correct_count = 0
        for step , (batch_X, batch_Y) in enumerate(train_loader):
            batch_X = batch_X.view(-1, 16*16)
            output = model(batch_X.float())
            loss = loss_func(output, batch_Y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            cls_pred = torch.max(output.data, 1)[1] # calculate max in row and return the index of max
            correct_count += (cls_pred == batch_Y).sum()

            if step % 20 == 0:
                # correct_count += (cls_pred == batch_Y).sum()
                print('Epoch: ', i+1, 'data_size: ', (step+1)*batch_Y.size(0), 'training_loss: %.4f' % loss.data.numpy(), 'Accuracy: %.2f' % (correct_count/(batch_Y.size(0)*(step+1))))
# torch.save(mlp_model.state_dict(), './datasets/mlp_model.pkl')


def testing(data_loader, model):
    correct_count = 0
    with torch.no_grad():
        for _, (set_X, set_Y) in enumerate(data_loader):
            set_X = set_X.view(-1, 16*16)
            output = model(set_X.float())
            cls_pred = torch.max(output.data, 1)[1]
            correct_count += (cls_pred == set_Y).sum()
    print('Accuracy on test dataset', float(correct_count)/float(set_Y.size(0)))
    # print(len(test_Y))
    

if __name__ == "__main__":  
    # train
    # training(train_loader, mlp_model, optim, loss_function)

    # test
    mlp_model.load_state_dict(torch.load('./datasets/mlp_model.pkl'))
    testing(test_loader, mlp_model)

