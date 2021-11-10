import numpy as np
import os
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import gc
import datetime
import argparse, sys


BASE_PATH = '/export/research26/cyclone/hansika/noc_data'
DIR = '64_nodes_100_c'
NO_OF_FILES = 39
No_OF_EPOCHS = 5


parser=argparse.ArgumentParser()

parser.add_argument('--base-path', help='base path')
parser.add_argument('--dir', help='specific directory')
parser.add_argument('--no-of-files', help='no of data files')
parser.add_argument('--no-of-epochs', help='no epochs for training')

args=parser.parse_args()

if args.base_path != None:
    BASE_PATH = args.base_path
if args.dir != None:
    DIR = args.dir 
if args.no_of_files!= None:
    NO_OF_FILES = int(args.no_of_files)
else:
    if DIR[-1] == 'c' or DIR[-2] == 'c':
        NO_OF_FILES = 39
    else:
        NO_OF_FILES = 41
if args.no_of_epochs!= None:
    No_OF_EPOCHS = int(args.no_of_epochs)

def print_and_write_to_file(filez, text1, text2=None):
    if text2 != None:
        text = str(text1) + str(text2)
    else:
        text = str(text1)
    filez.write(text)
    print(text)
    filez.write("\n")


filez = open(BASE_PATH + "/model_test_results/epoch_" + str(No_OF_EPOCHS) + "/" + DIR , 'a+')
print_and_write_to_file(filez, '----------------------------------------------------------')
print_and_write_to_file(filez, datetime.datetime.now())


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
number_of_files = NO_OF_FILES
print_and_write_to_file(filez,"No of flies : " + str(number_of_files))
print_and_write_to_file(filez,"Reading from : " + BASE_PATH + "/numpy_data_reduced/" + DIR )

for i in range(number_of_files):
     list_of_dataset.append(MyDataset(BASE_PATH + "/numpy_data_reduced/" + DIR + "/",i))

full_dataset = ConcatDataset(list_of_dataset)

print_and_write_to_file(filez,len(full_dataset))


classes = [label.item() for _, label in full_dataset]
print_and_write_to_file(filez,Counter(classes))

gc.collect()

## ---------------------------------------- CNN initialization ----------------------------- ##



# W1, W2, K1, K2 are hyper parameters that eventually needed training
W1 = 30
W2 = 10
K1= 2000
K2 = 1000

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

dir_of_model = DIR[0: DIR.rfind("_")+1]

dataset = torch.utils.data.DataLoader(full_dataset, batch_size=50, shuffle=True)
print_and_write_to_file(filez,"Testing with : " + BASE_PATH + "/models/epoch_" + str(No_OF_EPOCHS) + "/" + dir_of_model)
net = torch.load(BASE_PATH + "/models/epoch_" + str(No_OF_EPOCHS) + "/" + dir_of_model)

correct = 0
TP = 0
TN = 0
FP = 0
FN = 0
total = 0

with torch.no_grad():
    for data in dataset:
        X, y = data
        X = X.type(torch.FloatTensor)
        output = net(X.view(-1,1,2,450))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
                if y[idx] == 1:
                    TP +=1
                else:
                    TN +=1
            else:
                if y[idx] == 1:
                    FN +=1
                else:
                    FP +=1
            total += 1

print_and_write_to_file(filez,"Accuracy: ", round(correct/total, 3))  
print_and_write_to_file(filez,"TP: ", TP)  
print_and_write_to_file(filez,"TN: ", TN)  
print_and_write_to_file(filez,"FP: ", FP)  
print_and_write_to_file(filez,"FN: ", FN)  

print_and_write_to_file(filez,datetime.datetime.now())



