import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2 as cv


DIC = ["blood","famine","garbage","Post-war-ruins","others"]
IMG_PATH = './data/origin/blood/2.220-blood-girl_1001139f.jpg'
MODEL_PATH = './net.tar'
SALIENCY_TH = 0.0005
RED_TH = 128
BLUE_TH = 64
GREEN_TH = 64
IMAGE_SIZE = 64


class main():
    def __init__(self,img_path,model_path):
        self.img_path = img_path
        self.img = Image.open(img_path)
        self.img = self.img.resize((IMAGE_SIZE,IMAGE_SIZE))
        self.img = self.img.convert('RGB')
        self.net = torch.load(model_path)
        self.device = torch.device('cuda')
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.blur_points = []
        # print(self.img.shape)
    def predict(self):
        data = self.transform(self.img)
        data = torch.unsqueeze(data,0)
        data = data.to(self.device)
        # print(data)
        output = self.net(data)
        m = output.cpu().data.numpy()[0]
        m_exp = np.exp(m)
        m_exp_row_sum = m_exp.sum(axis=0)
        softmax = m_exp / m_exp_row_sum
        return softmax

    def getSaliency(self):
        data = self.transform(self.img)
        data = torch.unsqueeze(data, 0)
        data = data.to(self.device)
        data.requires_grad_()
    # img = torch.tensor([self.img]).type('torch.FloatTensor').cuda()
        output = self.net(data)
        _, predicted = torch.max(output.data, 1)
        # total += labels.size(0)
        predicted = predicted.cpu().numpy()
        output = output.gather(1, torch.LongTensor(predicted).to(self.device).view(-1, 1)).squeeze()
        output.backward(gradient = torch.Tensor([1.0]).to(self.device))
        saliency = data.grad

        # Convert 3d to 1d
        saliency = saliency.abs()
        saliency, _ = torch.max(saliency, dim=1)
        return torch.squeeze(saliency).cpu().numpy()


    def getBlur(self):
        saliency = self.getSaliency()
        img = cv.imread(self.img_path)
        h,w,_ = img.shape
        grad_points = np.where(saliency > SALIENCY_TH)
        self.blur_points = []
        filter = np.ones((IMAGE_SIZE, IMAGE_SIZE,3),dtype = np.uint8)
        filter[grad_points] = [0,0,0]
        origin = cv.imread(self.img_path)
        # origin = cv.resize(origin, (500, 500))
        filter = cv.resize(filter,(w,h))
        b,g,r = cv.split(origin)
        blue_points = np.where(b > BLUE_TH)
        green_points = np.where(g > GREEN_TH)
        filter[blue_points] = [1,1,1]
        filter[green_points] = [1,1,1]
        img *= filter
        cv.imshow("blur",img)
        cv.imshow("origin",origin)
        cv.waitKey()


    def mainFunc(self):
        predict = self.predict()
        label = np.argmax(predict,0)
        for i in range(len(predict)):
            print("%s类的概率为:%.5f%%" % (DIC[i],predict[i] * 100))
        # print(label)
        if label == 0:
            self.getBlur()

if __name__ == '__main__':
    t = main(IMG_PATH,MODEL_PATH)
    t.mainFunc()
