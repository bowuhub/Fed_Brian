import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 10
num_classes = 4
dir_path = "/root/Desktop/file/fl/dataset/dir/"


# Allocate data to users
def generate_tumor_2(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])
    # Load train and test data
    train_data = torchvision.datasets.ImageFolder(root='/root/Desktop/data/Brain_Tumor_2/train', transform=transform)
    test_data = torchvision.datasets.ImageFolder(root='/root/Desktop/data/Brain_Tumor_2/test', transform=transform)

    dataset_image = []
    dataset_label = []

    for i in range(len(train_data)):
        dataset_image.append(train_data[i][0].numpy())
        dataset_label.append(torch.tensor(train_data[i][1], dtype=torch.long))

    for i in range(len(test_data)):
        dataset_image.append(test_data[i][0].numpy())
        dataset_label.append(torch.tensor(test_data[i][1], dtype=torch.long))

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_tumor_2(dir_path, num_clients, num_classes, niid, balance, partition)
