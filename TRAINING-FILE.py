import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from MYRESNET import ResNet34
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image



TRAIN_PATH = './data/train.txt'
TEST_PATH = './data/test.txt'
TRAIN_LOG_PATH = './log/train.txt'
TEST_LOG_PATH = './log/test.txt'
MODEL_PATH = './net.tar'
BATCH_SIZE = 40
LR = 0.1
EPOCH = 30
CLASS_NUM = 5
IMAGE_SIZE = 64



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
        img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
        img = self.transforms(img)
        # img = img[np.newaxis,:,:]#add dimention
        label = torch.LongTensor([self.label_lists[index]])
        # print(self.img_lists[index],img.shape)
        return img,label

    def __len__(self):
        return len(self.label_lists)



class MAIN():
    def __init__(self,train_path,test_path):
        self.device = torch.device('cuda')
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_set = MyDataset(train_path,transforms = self.transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_set,batch_size = BATCH_SIZE,shuffle = True)
        self.test_set = MyDataset(test_path,transforms = self.transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_set,batch_size = BATCH_SIZE,shuffle = True)
        self.net = ResNet34(num_classes = CLASS_NUM).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(),lr = LR,momentum = 0.9,weight_decay = 5e-4)
        print(len(self.train_loader))



    def test(self,epoch):
        correct = 0.0
        total = 0.0
        for i, data in enumerate(self.test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy().flatten()
            total += labels.shape[0]
            correct += np.sum(predicted == labels)
            print('[epoch:%d, iter:%d] | Test acc: %.3f%% ' % (
                epoch + 1, i, 100. * correct / total))

        with open(TEST_LOG_PATH,'a') as f:
            f.write(str(epoch) + ':' + str(100. * correct / total) + '\n')


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
                # total += labels.size(0)
                predicted = predicted.cpu().numpy()
                labels = labels.cpu().numpy().flatten()
                # print(inputs.grad)
                print(predicted)
                print(labels.flatten())
                total += labels.shape[0]
                correct += np.sum(predicted == labels)
                # correct += predicted.eq(labels.data).cpu().sum()
                # # print(correct.shape)
                # # print("correct:" + str(correct))
                print('[epoch:%d, iter:%d] Loss: %.03f | Train acc: %.3f%% ' % (
                epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
            with open(TRAIN_LOG_PATH,'a') as f:
                f.write(str(epoch) + ':' + str(100. * correct / total) + '\n')
            self.test(epoch)


    def mainFunc(self):
        self.train()
        torch.save(self.net, MODEL_PATH)

if __name__ == '__main__':
    t = MAIN(TRAIN_PATH,TEST_PATH)
    t.mainFunc()