import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from MYRESNET import ResNet18
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image

TRAIN_PATH = './data/train.txt'
BATCH_SIZE = 20
LR = 0.1
EPOCH = 50
class MyDataset(Dataset):
    def __init__(self,txt_path,transforms):
        self.transforms = transforms
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.img_lists = [i.split(" ")[0] for i in lines]

            self.label_lists = [int(i.split(" ")[1][:-1]) for i in lines]
            print(self.img_lists)
            # print(self.label_lists)
    def __getitem__(self, index):
        img = Image.open(self.img_lists[index])
        img = img.convert('RGB')
        img = img.resize((32,32))
        img = self.transforms(img)
        # img = img[np.newaxis,:,:]#add dimention
        label = torch.LongTensor([self.label_lists[index]])
        # print(self.img_lists[index],img.shape)
        return img,label

    def __len__(self):
        return len(self.label_lists)



class MAIN():
    def __init__(self,train_path):
        self.device = torch.device('cuda')
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_set = MyDataset(train_path,transforms = self.transform_train)
        self.train_loader = torch.utils.data.DataLoader(self.train_set,batch_size = BATCH_SIZE,shuffle = True)
        self.net = ResNet18(num_classes = 5).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(),lr = LR,momentum = 0.9,weight_decay = 5e-4)
        print(len(self.train_loader))


    def train(self):
        for epoch in range(EPOCH):
            self.net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, data in enumerate(self.train_loader):
                length = len(self.train_loader)
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                # print(outputs)
                # print(outputs.shape)
                loss = self.criterion(outputs, labels.flatten())
                # print(labels.flatten())
                # print(torch.max(labels, 1)[1])
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                print(predicted)
                print(labels)
                total += labels.size(0)
                # print(labels)
                correct += predicted.eq(labels.data).cpu().sum()
                # print(correct.shape)
                # print("correct:" + str(correct))
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' % (
                epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))


    def mainFunc(self):
        self.train()


if __name__ == '__main__':
    t = MAIN(TRAIN_PATH)
    t.mainFunc()