import cv2 as cv
import os
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import random
import numpy as np

FOLD_ROOT = './data/others'
#操作类型
#旋转
OPERATION_ROTATION = True
#颜色
OPERATION_COLOR = True
#高斯噪声
OPERATION_NOISY = True
#随机删除比例
DELETE_SCALE = 0.9

class main():
    def __init__(self,root):
        self.root = root
        self.img_paths = [FOLD_ROOT + '/' + i for i in os.listdir(FOLD_ROOT)]

    def deleteFile(self):
        delete_list = random.sample(self.img_paths,int(DELETE_SCALE * len(self.img_paths)))
        for path in delete_list:
            os.remove(path)



    def compressImages(self):
        for image_path in self.img_paths:
            print(image_path)
            img = cv.imread(image_path)
            h , w , c = img.shape

            res = cv.resize(img,(h // 2, w // 2),interpolation=cv.INTER_AREA)
            cv.imwrite(image_path,res)

    def randomRotation(image, mode=Image.BICUBIC):
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode)


    def dataAugmentation(self):
        pass
    def mainFunc(self):
        # self.compressImages()
        self.deleteFile()





if __name__ == '__main__':
    # print(os.getcwd())
    t = main(FOLD_ROOT)
    t.mainFunc()