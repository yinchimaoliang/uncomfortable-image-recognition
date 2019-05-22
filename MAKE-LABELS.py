import os
import random

DATASET_PATH = './data'
TRAIN_PATH = './data/train.txt'
TEST_PATH = './data/test.txt'
TEST_SCALE = 0.2



class main():
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path



    def readData(self):
        self.folds = os.listdir(self.dataset_path)

        self.data = []

        for fold_name in self.folds:
            path = DATASET_PATH + '/' + fold_name
            print(path)
            self.data.append([DATASET_PATH + path[6:] + '/' + i for i in os.listdir(path)])

    def makeLabels(self):
        for i in range(len(self.data)):
            print(len(self.data[i]))
            for j in range(len(self.data[i])):
                choice = random.random()
                #随机生成训练集和测试集
                if choice > 0.2:
                    with open(TRAIN_PATH, 'a') as f:
                        f.write(self.data[i][j] + ' ' + str(i) + '\n')
                else:
                    with open(TEST_PATH, 'a') as f:
                        f.write(self.data[i][j] + ' ' + str(i) + '\n')

    def mainFunc(self):
        self.readData()
        # print(self.data)
        self.makeLabels()
        # for i, j, k in os.walk(self.dataset_path):
        #     print(i, j, k)



if __name__ == '__main__':
    t = main(DATASET_PATH)
    t.mainFunc()