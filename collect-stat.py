import numpy as np
import os
import torch
import torch
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
number_of_files = 39
from torch.utils.data import ConcatDataset

for i in range(number_of_files):
     list_of_dataset.append(MyDataset("/export/research26/cyclone/hansika/noc_data/numpy_data_reduced/64_nodes_95_c/",i))

full_dataset = ConcatDataset(list_of_dataset)

print(len(full_dataset))

# data_set = MyDataset("/home/hansika/gem5/gem5/scripts/numpy_data_reduced/64_nodes/",1)
# train_data_set, test_data_set = torch.utils.data.random_split(data_set, [300, 96]) 
# train_data_set, test_data_set = torch.utils.data.random_split(full_dataset, [10128, 6000]) 


classes = [label.item() for _, label in full_dataset]
print(Counter(classes))

gc.collect()



