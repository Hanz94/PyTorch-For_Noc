import numpy as np
import os
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import gc
import datetime


print(datetime.datetime.now())

class MyDataset(Dataset):
    def __init__(self, data_dir, file_index):
        self.data_dir = data_dir
        self.file_index = file_index
        y = np.load(self.data_dir + "Y/" + str(self.file_index) + ".npy", allow_pickle=True)
        self.y = torch.from_numpy(y)
    
    def __getitem__(self, index):
        x = np.load(self.data_dir + "X/" + str(self.file_index) + ".npy", allow_pickle=True, mmap_mode='r')
        x = torch.from_numpy(x)
        return [x[index], self.y[index]]
    
    def __len__(self):
        return len(self.y)


list_of_dataset = []
number_of_files = 38
from torch.utils.data import ConcatDataset

for i in range(number_of_files):
     list_of_dataset.append(MyDataset("/cise/homes/hansikam.lokukat/64_nodes_100_c/",i))

full_dataset = ConcatDataset(list_of_dataset)

print(len(full_dataset))

# data_set = MyDataset("/home/hansika/gem5/gem5/scripts/numpy_data_reduced/64_nodes/",1)
# train_data_set, test_data_set = torch.utils.data.random_split(data_set, [300, 96]) 
# train_data_set, test_data_set = torch.utils.data.random_split(full_dataset, [10128, 6000]) 


classes = [label.item() for _, label in full_dataset]
print(Counter(classes))

gc.collect()

# print(data_set[10])

## ---------------------------------------- CNN initialization ----------------------------- ##



# W1, W2, K1, K2 are hyper parameters that eventually needed training
W1 = 30
W2 = 10
K1= 2000
K2 = 1000

#dummy data to try the NN ( 2 arrays of size 450)
# dummy = torch.randn(2,450).view(-1,1,2,450)

# represents the whole CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, K1, (2,W1), stride=(2, 1))
        self.pool1 = nn.MaxPool2d((1, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(K1, K2, (1,W2), stride=(1, 1))
        self.pool2 = nn.MaxPool2d((1, 5), stride=(1, 1))
        
        self.fc1 = nn.Linear(1000*404, 3000) # need to automate arriving at this number (1000*254)
        self.fc2 = nn.Linear(3000, 800) 
        self.fc3 = nn.Linear(800,100)
        self.fc4 = nn.Linear(100,2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = x.view(-1, 1000*404)    
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        # return torch.sigmoid(x)
    
# ------------------- Training the CNN ------------------------------------- ##
# For now this code is only to show the structure, I need to add data preparation and modify code accordingly.


dataset = torch.utils.data.DataLoader(full_dataset, batch_size=50, shuffle=True)
net = torch.load("/cise/homes/hansikam.lokukat/64_nodes_100_model")

correct = 0
total = 0

with torch.no_grad():
    for data in dataset:
        X, y = data
        X = X.type(torch.FloatTensor)
        output = net(X.view(-1,1,2,450))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))  

print(datetime.datetime.now())



