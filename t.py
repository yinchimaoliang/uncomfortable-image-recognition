import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import argparse
from MYRESNET import ResNet18
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
from PIL import Image


IMG_PATH = './data/blood/1.accident-arm-bleeding-blood-bloody-body-cut-gash-health-care-injured-D9ENMR.jpg'
MODEL_PATH = './net.tar'
BATCH_SIZE = 40
TH = 0.001

class MyDataset(Dataset):
    def __init__(self,img_path,transforms):
        self.transforms = transforms
        self.img_path = img_path
            # print(self.label_lists)
    def __getitem__(self, index):
        img = Image.open(self.img_path)
        img = img.convert('RGB')
        img = img.resize((32,32))
        img = self.transforms(img)
        return img

    def __len__(self):
        return 1






class main():
    def __init__(self,img_path,model_path):
        self.img = Image.open(img_path)
        self.img = self.img.resize((32,32))
        self.img = self.img.convert('RGB')
        self.net = torch.load(model_path)
        self.device = torch.device('cuda')
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_set = MyDataset(IMG_PATH, transforms=self.transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=BATCH_SIZE, shuffle=True)
        self.blur_points = []
        # print(self.img.shape)
    def predict(self):
        for i, data in enumerate(self.train_loader):
            data = data.to(self.device)
            output = self.net(data)
            _, predicted = torch.max(output.data, 1)
            # total += labels.size(0)
            predicted = predicted.cpu().numpy()
        return predicted

    def getSaliency(self):
        for i, data in enumerate(self.train_loader):
            data = data.to(self.device)
            data.requires_grad_()
        # img = torch.tensor([self.img]).type('torch.FloatTensor').cuda()
            output = self.net(data)
            print(output)
            _, predicted = torch.max(output.data, 1)
            # total += labels.size(0)
            predicted = predicted.cpu().numpy()
            output = output.gather(1, torch.LongTensor(predicted).to(self.device).view(-1, 1)).squeeze()
            output.backward(gradient = torch.Tensor([1.0]).to(self.device))
            saliency = data.grad

            # Convert 3d to 1d
            saliency = saliency.abs()
            saliency, _ = torch.max(saliency, dim=1)
            return saliency


    def getBlur(self):
        saliency = self.getSaliency()
        for row in range(len(saliency[0])):
            for col in range(len(saliency[0][row])):
                if saliency[0][row][col].data.cpu().numpy() > TH:
                    self.blur_points.append([row,col])

        print(self.blur_points)
    # def get

    # def mainFunc(self):
    #     print(self.img)


if __name__ == '__main__':
    t = main(IMG_PATH,MODEL_PATH)
    print(t.predict())
    t.getBlur()
