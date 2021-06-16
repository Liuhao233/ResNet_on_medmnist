import os
import sys
from torchvision.transforms.transforms import Resize
import torchvision
from tqdm import trange
from tqdm import tqdm
from skimage.util import montage
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from info import INFO
from resnet18 import ResNet18
from resnet_50 import ResNet50
data_flag = "retinamnist"
download = True
input_root = 'tmp_data/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
BATCH_SIZE = 128
lr = 0.01

flag_to_class = {
    "pathmnist": PathMNIST,
    "chestmnist": ChestMNIST,
    "dermamnist": DermaMNIST,
    "octmnist": OCTMNIST,
    "pneumoniamnist": PneumoniaMNIST,
    "retinamnist": RetinaMNIST,
    "breastmnist": BreastMNIST,
    "organmnist_axial": OrganMNISTAxial,
    "organmnist_coronal": OrganMNISTCoronal,
    "organmnist_sagittal": OrganMNISTSagittal,
}

DataClass = flag_to_class[data_flag]

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
'''
data_transform = transforms.Compose([
    Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])
'''
data_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5]),
])

train_dataset = DataClass(root=input_root, split='train', transform=data_transform, download=download)
test_dataset = DataClass(root=input_root, split='test', transform=data_transform, download=download)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)


resnet_18= ResNet50().to(device)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(resnet_18.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
for epoch in range(NUM_EPOCHS):
    print('\nEpoch: %d' % (epoch + 1))
    resnet_18.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    i=0
    for data in tqdm(train_loader):
                    # 准备数据
        length = len(train_loader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # forward + backward
        outputs = resnet_18(inputs)
        labels=labels.squeeze()
        labels = torch.tensor(labels, dtype=torch.long)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        i=i+1
    print('[epoch:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, sum_loss / (i + 1), 100. * correct / total))
                    
