import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2 as cv



IMG_PATH = './data/blood/1.accident-arm-bleeding-blood-bloody-body-cut-gash-health-care-injured-D9ENMR.jpg'
MODEL_PATH = './net.tar'
SALIENCY_TH = 0.001
RED_TH = 128
BLUE_TH = 64
GREEN_TH = 64
IMAGE_SIZE = 32


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
        # print(output)
        _, predicted = torch.max(output.data, 1)
        # print(predicted)
        # total += labels.size(0)
        predicted = predicted.cpu().numpy()
        return predicted

    def getSaliency(self):
        data = self.transform(self.img)
        data = torch.unsqueeze(data, 0)
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
        return torch.squeeze(saliency).cpu().numpy()


    def getBlur(self):
        saliency = self.getSaliency()
        img = cv.imread(self.img_path)
        h,w,_ = img.shape
        print(h,w)
        # print(img.shape)
        b, g, r = cv.split(img)
        grad_points = np.array(np.where(saliency > SALIENCY_TH))
        # print(grad_points)
        red_points = np.array(np.where(r > RED_TH)).T
        blue_points = np.array(np.where(b < BLUE_TH)).T
        green_points = np.array(np.where(b < GREEN_TH)).T
        grad_points = grad_points.T
        self.blur_points = []
        filter = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3),dtype = np.uint8)
        for i in grad_points:
            if i in red_points and i in blue_points and i in grad_points:
                filter[i[0],i[1],:] = [0,0,0]
                self.blur_points.append(i)
        # print(red_points)

        origin = cv.imread(self.img_path)
        origin = cv.resize(origin, (500, 500))
        filter = cv.resize(filter,(w,h))
        # print(img.type)
        img *= filter
        img = cv.resize(img,(500,500))
        cv.imshow("test",img)
        cv.imshow("t",origin)
        cv.waitKey()


if __name__ == '__main__':
    t = main(IMG_PATH,MODEL_PATH)
    print(t.predict())
    t.getBlur()
