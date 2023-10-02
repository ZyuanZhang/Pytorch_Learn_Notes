# 参考: https://github.com/JansonYuan/Pytorch-Camp/
# 数据集已经划分完毕，其中:
# Train-Cats: 279; Train-Dogs: 278
# Valid-Cats: 35; Valid-Dogs: 35
# Test-Cats: 35; Test-Dogs: 35

# 可以直接将 "./cat_and_dog_images/" 和 "./rmb_split/" 进行解压即可直接运行该脚本

import random
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import transforms

import torch.optim as optim
from torch.optim import lr_scheduler

import torch

import numpy as np
import matplotlib.pyplot as plt




# -------- 构建数据加载器 --------
catdog_label = {"cats":0, "dogs":1} # cat&dog
#rmb_label = {"1":0, "100":1} # RMB

class CatsDogsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.label_name = {"cats":0, "dogs":1} # cat&dog
        #self.label_name = rmb_label # RMB
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.data_info)
    
    @staticmethod
    def get_img_info(data_dir):
        data_info = []
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith(".jpg"), img_names))

                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = catdog_label[sub_dir] # cat&dog
                    #label = rmb_label[sub_dir] # RMB
                    data_info.append((path_img, int(label)))
            
        return data_info




# -------- 定义模型 --------
class LeNet(nn.Module):
    def __init__(self, classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
    def initialize_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()



def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed()

# -------- 设置参数 --------
MAX_EPOCH = 10
BATCH_SIZE = 16
LR = 0.01
log_interval = 35 # cat&dog
#log_interval = 10 # RMB
val_interval = 1




# -------- 1/5 数据 --------
train_dir = './cat_and_dog_images/train/' # cat&dog
val_dir = "./cat_and_dog_images/valid/" #cat&dog
#test_dir = "./cat_and_dog_images/test/"

#train_dir = "./rmb_split/train/" # RMB
#val_dir = "./rmb_split/valid/" # RMB


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

val_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

train_data = CatsDogsDataset(data_dir=train_dir, transform=train_transform)
val_data = CatsDogsDataset(data_dir=val_dir, transform=val_transform)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE)
print(len(train_loader))
print(len(val_loader))



# -------- 2/5 模型 --------
net = LeNet(classes=2)
net.initialize_weights()




# -------- 3/5 损失函数 --------
criterion = nn.CrossEntropyLoss()




# -------- 4/5 优化器 --------
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
schedular = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)




# -------- 5/5 训练 --------
train_curve, val_curve = [], []

for epoch in range(MAX_EPOCH):
    
    loss_mean = 0.
    correct = 0.
    total = 0.

    # train model
    net.train()
    for i, data in enumerate(train_loader):
        
        # forward
        inputs, labels = data
        outputs = net(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # count num of corrects
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()

        # print info of training
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Train:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc: {:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.
    
    schedular.step() # update learning rate

    # val model
    if (epoch+1) % val_interval == 0:
        loss_val = 0.
        correct_val = 0.
        total_val = 0.

        net.eval()
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().numpy()

                loss_val += loss.item()

            loss_val_epoch = loss_val / len(val_loader)
            val_curve.append(loss_val_epoch)
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc: {:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(val_loader), loss_val_epoch, correct_val / total_val))


train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1, len(val_curve)+1) * train_iters*val_interval - 1
valid_y = val_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()




