import cv2 as cv
import os
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import random
import numpy as np

FOLD_ROOT = './data/blood'
OUTPUT_ROOT = './data/processed/blood'
#操作类型
#旋转
OPERATION_ROTATION = True
ROTATION_NUM = 5
#颜色
OPERATION_COLOR = False
COLOR_NUM = 5
#高斯噪声
OPERATION_NOISY = True
NOISY_NUM = 5
#随机删除比例
DELETE_SCALE = 0.9

class main():
    def __init__(self,root):
        self.root = root
        self.img_paths = [FOLD_ROOT + '/' + i for i in os.listdir(FOLD_ROOT)]
        self.img_paths = self.img_paths
        self.crop_image = lambda img, x0, y0, w, h: img[y0:y0 + h, x0:x0 + w]
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

    def rotate_image(self,img, angle, crop):
        h, w = img.shape[:2]  # 旋转角度的周期是360° angle %= 360 # 用OpenCV内置函数计算仿射矩阵
        M_rotate = cv.getRotationMatrix2D((w/2, h/2), angle, 1) # 得到旋转后的图像
        img_rotated = cv.warpAffine(img, M_rotate, (w, h)) # 如果需要裁剪去除黑边
        if crop:
            angle_crop = angle % 180 # 对于裁剪角度的等效周期是180°
            if angle_crop > 90: # 并且关于90°对称
                angle_crop = 180 - angle_crop
            theta = angle_crop * np.pi / 180.0 # 转化角度为弧度
            hw_ratio = float(h) / float(w) # 计算高宽比
            tan_theta = np.tan(theta) # 计算裁剪边长系数的分子项
            numerator = np.cos(theta) + np.sin(theta) * tan_theta
            r = hw_ratio if h > w else 1 / hw_ratio # 计算分母项中和宽高比相关的项
            denominator = r * tan_theta + 1	# 计算分母项
            crop_mult = numerator / denominator # 计算最终的边长系数
            w_crop = int(round(crop_mult*w))	# 得到裁剪区域
            h_crop = int(round(crop_mult*h))
            x0 = int((w-w_crop)/2)
            y0 = int((h-h_crop)/2)
            img_rotated = self.crop_image(img_rotated, x0, y0, w_crop, h_crop)
        return img_rotated




    #随机旋转
    def random_rotate(self,img, angle_vari, p_crop):
        angle = np.random.uniform(-angle_vari, angle_vari)

        crop = False if np.random.random() > p_crop else True
        return self.rotate_image(img, angle, crop)

    #明暗对比度变化
    def hsv_transform(self,img, hue_delta, sat_mult, val_mult):
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float)

        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        img_hsv[:, :, 1] *= sat_mult
        img_hsv[:, :, 2] *= val_mult
        img_hsv[img_hsv > 255] = 255
        return cv.cvtColor(np.round(img_hsv).astype(np.uint8), cv.COLOR_HSV2BGR)



    #
    def random_hsv_transform(self,img, hue_vari, sat_vari, val_vari):
        hue_delta = np.random.randint(-hue_vari, hue_vari)

        sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
        val_mult = 1 + np.random.uniform(-val_vari, val_vari)
        return self.hsv_transform(img, hue_delta, sat_mult, val_mult)

    def salt(self,img):
        # 循环添加n个椒盐
        n = img.shape[1] * img.shape[0] // 50
        for k in range(n):
            # 随机选择椒盐的坐标
            i = int(np.random.random() * img.shape[1])
            j = int(np.random.random() * img.shape[0])
            # 如果是灰度图
            if img.ndim == 2:
                img[j, i] = 255
            # 如果是RBG图片
            elif img.ndim == 3:
                img[j, i, 0] = 255
                img[j, i, 1] = 255
                img[j, i, 2] = 255
        return img

    def noiseing(self,img):
        param = 30
        grayscale = 256
        w = img.shape[1]
        h = img.shape[0]
        newimg = np.zeros((h, w), np.uint8)

        for x in range(0, h):
            for y in range(0, w, 2):
                r1 = np.random.random_sample()
                r2 = np.random.random_sample()
                z1 = param * np.cos(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
                z2 = param * np.sin(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))

                fxy = int(img[x, y] + z1)
                fxy1 = int(img[x, y + 1] + z2)
                if fxy < 0:
                    fxy_val = 0
                elif fxy > grayscale - 1:
                    fxy_val = grayscale - 1
                else:
                    fxy_val = fxy
                if fxy1 < 0:
                    fxy1_val = 0
                elif fxy1 > grayscale - 1:
                    fxy1_val = grayscale - 1
                else:
                    fxy1_val = fxy1
                newimg[x, y] = fxy_val
                newimg[x, y + 1] = fxy1_val

        return newimg

    def addGaussianNoise(self,image, percetage):
        G_Noiseimg = image.copy()
        w = image.shape[1]
        h = image.shape[0]
        G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
        for i in range(G_NoiseNum):
            temp_x = np.random.randint(0, h)
            temp_y = np.random.randint(0, w)
            G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
        return G_Noiseimg


    def dataAugmentation(self):
        count = 0
        for img_path in self.img_paths:
            img = cv.imread(img_path)
            print(img_path)
            if OPERATION_ROTATION:
                for i in range(ROTATION_NUM):
                    # img_processed = self.addGaussianNoise(img,0.5)
                    img_processed = self.random_rotate(img,90,1)
                    cv.imwrite(OUTPUT_ROOT + '/' + str(count) + ".jpg",img_processed)
                    count += 1
                    print("finish %d images" % count)

            if OPERATION_COLOR:
                for i in range(COLOR_NUM):
                    img_processed = self.random_hsv_transform(img,1,1,1)
                    cv.imwrite(OUTPUT_ROOT + '/' + str(count) + ".jpg", img_processed)
                    count += 1
                    print("finish %d images" % count)

            if OPERATION_ROTATION:
                for i in range(NOISY_NUM):
                    img_processed = self.salt(img)
                    cv.imwrite(OUTPUT_ROOT + '/' + str(count) + ".jpg", img_processed)
                    count += 1
                    print("finish %d images" % count)

        # cv.imshow("test",img)
        # cv.waitKey()


    def mainFunc(self):
        self.dataAugmentation()
        # self.compressImages()
        # self.deleteFile()





if __name__ == '__main__':
    # print(os.getcwd())
    t = main(FOLD_ROOT)
    t.mainFunc()