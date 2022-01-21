import numpy as np
import os
from tqdm import tqdm

dataset_name = 'dsads'
class_num = 19
dataset = np.load('data/{}.npy'.format(dataset_name), allow_pickle=True).item()
print(dataset['y_train'].shape)
subject_num = len(dataset['y_train'])
val_idx = 0
test_idx = 1
split_file_name_list = ['train', 'val', 'test']
split_file_array_list = [[], [], []]

dataset_folder = 'data/{}'.format(dataset_name)
if not os.path.exists(dataset_folder):
    os.mkdir(dataset_folder)
for c in range(19):
    folder = '{}/{}'.format(dataset_folder, c)
    if not os.path.exists(folder):
        os.mkdir(folder)
for i in range(subject_num):
    X_mat = np.r_[dataset['X_train'][i], dataset['X_test'][i]]
    y_mat = np.r_[dataset['y_train'][i], dataset['y_test'][i]]
    for j in tqdm(range(len(y_mat))):
        y = int(y_mat[j])
        file_name = '{}/{}.npy'.format(y, j)
        np.save('{}/{}'.format(dataset_folder, file_name), X_mat[j])
        set_idx = 0
        if i == val_idx:
            set_idx = 1
        elif i == test_idx:
            set_idx = 2

        split_file_array_list[set_idx].append(np.array([file_name, str(y)]))

split_folder = 'split/{}'.format(dataset_name)
if not os.path.exists(split_folder):
    os.mkdir(split_folder)
for set_idx in range(3):
    split_array = np.r_[split_file_array_list[set_idx]]
    np.savetxt('{}/{}.csv'.format(split_folder, split_file_name_list[set_idx]),
               split_array, delimiter=",", fmt="%s")


