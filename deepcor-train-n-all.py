import numpy as np
import os
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
No_OF_EPOCHS = 5
NO_OF_FLITS = 250
NOISE_RATIO = 0
MAIN_DIR = "/actual_0"

parser = argparse.ArgumentParser()

parser.add_argument('--base-path', help='base path')
parser.add_argument('--no-of-epochs', help='no epochs for training')
parser.add_argument('--no-of-flits', help='IFD length for training')
parser.add_argument('--noise-ratio', help='noise ratio of the collected data, defaults to 0')

args = parser.parse_args()

if args.base_path != None:
    BASE_PATH = args.base_path
if args.no_of_epochs != None:
    No_OF_EPOCHS = int(args.no_of_epochs)
if args.no_of_flits != None:
    NO_OF_FLITS = int(args.no_of_flits)
if args.noise_ratio != None:
    NOISE_RATIO = int(args.noise_ratio)
    MAIN_DIR = "/actual_" + str(NOISE_RATIO)


def print_and_write_to_file(filez, text1, text2=None):
    if text2 != None:
        text = str(text1) + str(text2)
    else:
        text = str(text1)
    filez.write(text)
    print(text)
    filez.write("\n")


filez = open(BASE_PATH + "/model_train_results/epoch_" + str(No_OF_EPOCHS) + MAIN_DIR, 'a+')
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
ben_order = ["FFT", "FMM", "LU", "BARNES", "RADIX", "FFT"]

for x in range(5):
    dir_path = "64_nodes__" + ben_order[x] + "_" + ben_order[x+1] + "_" + str(NOISE_RATIO) + "_" + str(NO_OF_FLITS)
    count_path = BASE_PATH + "/numpy_data_reduced/" + dir_path + "/" + "Y"
    number_of_files = len([name for name in os.listdir(count_path) if os.path.isfile(os.path.join(count_path, name))])
    print_and_write_to_file(filez, "No of flies for" + ben_order[x] + "_" + ben_order[x+1] + ": " + str(number_of_files))
    print_and_write_to_file(filez, "Reading from : " + BASE_PATH + "/numpy_data_reduced/" + dir_path)
    for i in range(number_of_files):
        list_of_dataset.append(MyDataset(BASE_PATH + "/numpy_data_reduced/" + dir_path + "/", i))

full_dataset = ConcatDataset(list_of_dataset)

len_full = len(full_dataset)
print_and_write_to_file(filez, len_full)

train_data_set, test_data_set = torch.utils.data.random_split(full_dataset, [len_full - (len_full // 3), len_full // 3])

train_classes = [label.item() for _, label in train_data_set]
print_and_write_to_file(filez, Counter(train_classes))

test_classes = [label.item() for _, label in test_data_set]
print_and_write_to_file(filez, Counter(test_classes))

gc.collect()

## ---------------------------------------- CNN initialization ----------------------------- ##


# W1, W2, K1, K2 are hyper parameters that eventually needed training
W1 = 5
W2 = 30
K1 = 1000
K2 = 2000


# represents the whole CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, K1, (2, W1), stride=(2, 1))
        self.pool1 = nn.MaxPool2d((1, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(K1, K2, (1, W2), stride=(1, 1))
        self.pool2 = nn.MaxPool2d((1, 5), stride=(1, 1))

        x = torch.randn(2, NO_OF_FLITS).view(-1, 1, 2, NO_OF_FLITS)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 3000)
        self.fc2 = nn.Linear(3000, 800)
        self.fc3 = nn.Linear(800, 100)
        self.fc4 = nn.Linear(100, 2)

    def convs(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            print_and_write_to_file(filez, self._to_linear)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ------------------- Training the CNN ------------------------------------- ##
# For now this code is only to show the structure, I need to add data preparation and modify code accordingly.

net = Net()

print_and_write_to_file(filez, "W1 : ", W1)
print_and_write_to_file(filez, "W2 : ", W2)
print_and_write_to_file(filez, "K1 : ", K1)
print_and_write_to_file(filez, "K2 : ", K2)

isTraining = True
if isTraining:

    BATCH_SIZE = 50
    EPOCHS = No_OF_EPOCHS
    learning_rate = 0.0001

    print_and_write_to_file(filez, "No of epochs : ", EPOCHS)
    print_and_write_to_file(filez, "Batch size : ", BATCH_SIZE)
    print_and_write_to_file(filez, "Learning rate : ", learning_rate)

    trainset = torch.utils.data.DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True)
    testset = torch.utils.data.DataLoader(test_data_set, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    for epoch in range(EPOCHS):
        for data in trainset:
            X, y = data
            net.zero_grad()
            X = X.type(torch.FloatTensor)
            output = net(X.view(-1, 1, 2, NO_OF_FLITS))
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
        print_and_write_to_file(filez, loss)

    torch.save(net, BASE_PATH + "/models_new/epoch_" + str(No_OF_EPOCHS) + "/" + MAIN_DIR)
    print_and_write_to_file(filez,
                            "Model Saved as : " + BASE_PATH + "/models_new/epoch_" + str(No_OF_EPOCHS) + "/" + MAIN_DIR)

    correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total = 0

    with torch.no_grad():
        for data in testset:
            X, y = data
            X = X.type(torch.FloatTensor)
            output = net(X.view(-1, 1, 2, NO_OF_FLITS))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                    if y[idx] == 1:
                        TP += 1
                    else:
                        TN += 1
                else:
                    if y[idx] == 1:
                        FN += 1
                    else:
                        FP += 1
                total += 1

    print_and_write_to_file(filez, "Accuracy: ", round(correct / total, 3))
    print_and_write_to_file(filez, "TP: ", TP)
    print_and_write_to_file(filez, "TN: ", TN)
    print_and_write_to_file(filez, "FP: ", FP)
    print_and_write_to_file(filez, "FN: ", FN)

print_and_write_to_file(filez, datetime.datetime.now())
filez.close()
