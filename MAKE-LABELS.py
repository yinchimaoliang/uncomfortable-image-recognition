import os
import random


ORIGIN_DATASET_PATH = './data/origin'
PROCESSED_DATASET_PATH = './data/processed'
TRAIN_PATH = './data/train.txt'
TEST_PATH = './data/test.txt'
TEST_SCALE = 0.2
DIC = {"blood":"0","famine":"1","garbage":"2","Post-war-ruins":"3","others":"4"}


class main():
    def __init__(self,origin_dataset_path,processed_data_path):
        self.origin_dataset_path = origin_dataset_path
        self.processed_dataset_path = processed_data_path



    def readData(self,dataset_path):
        self.folds = os.listdir(dataset_path)
        self.data = []

        for fold_name in self.folds:
            path = dataset_path + '/' + fold_name
            print(path)
            self.data.append([dataset_path + '/' + fold_name + '/' + i for i in os.listdir(path)])

    def makeLabels(self,test = True):
        for i in range(len(self.data)):
            print(len(self.data[i]))
            for j in range(len(self.data[i])):
                if test:
                    choice = random.random()
                    #随机生成训练集和测试集
                    if choice > 0.2:
                        with open(TRAIN_PATH, 'a') as f:
                            f.write(self.data[i][j] + ' ' + DIC[self.folds[i]] + '\n')
                    else:
                        with open(TEST_PATH, 'a') as f:
                            f.write(self.data[i][j] + ' ' + DIC[self.folds[i]] + '\n')

                else:
                    with open(TRAIN_PATH, 'a') as f:
                        f.write(self.data[i][j] + ' ' + DIC[self.folds[i]] + '\n')

    def mainFunc(self):
        self.readData(self.origin_dataset_path)
        # print(self.data)
        self.makeLabels(True)

        self.readData(self.processed_dataset_path)
        self.makeLabels(False)
        # for i, j, k in os.walk(self.dataset_path):
        #     print(i, j, k)



if __name__ == '__main__':
    t = main(ORIGIN_DATASET_PATH,PROCESSED_DATASET_PATH)
    t.mainFunc()