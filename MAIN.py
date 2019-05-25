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


IMG_PATH = './data/others/59315584_332882004061973_8733823118118270299_n.jpg'
MODEL_PATH = './net.tar'
BATCH_SIZE = 40

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
        # print(self.img.shape)
    def predict(self):
        for i, data in enumerate(self.train_loader):
            data = data.to(self.device)
        # img = torch.tensor([self.img]).type('torch.FloatTensor').cuda()
            output = self.net(data)
            print(output)
            _, predicted = torch.max(output.data, 1)
            # total += labels.size(0)
            predicted = predicted.cpu().numpy()
        return predicted

    def compute_saliency_maps(X, y, model):
        # Forward pass
        scores = model(X)

        # Correct class scores
        scores = scores.gather(1, y.view(-1, 1)).squeeze()

        # Backward pass
        # Note: scores is a tensor here, need to supply initial gradients of same tensor shape as scores.
        scores.backward(torch.ones(scores.size()))

        saliency = X.grad

        # Convert 3d to 1d
        saliency = saliency.abs()
        saliency, _ = torch.max(saliency, dim=1)
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return saliency


    def mainFunc(self):
        print(self.img)


if __name__ == '__main__':
    t = main(IMG_PATH,MODEL_PATH)
    print(t.predict())