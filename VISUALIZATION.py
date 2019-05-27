import matplotlib.pyplot as plt

LOG_PATH = './log'



class main():
    def __init__(self):
        self.test_file = LOG_PATH + '/' + 'test.txt'
        self.train_file = LOG_PATH + '/' + 'train.txt'


        self.train_log = []
        self.test_log = []


    def getData(self):
        with open(self.train_file,'r') as f:
            lines = f.readlines()
            for line in lines:
                epoch,rate = line.split(':')
                self.train_log.append(rate[:-2])

        with open(self.test_file,'r') as f:
            lines = f.readlines()
            for line in lines:
                epoch,rate = line.split(':')
                self.test_log.append(rate[:-2])

        # print(self.train_log,self.test_log)






    def mainFunc(self):
        self.getData()



if __name__ == '__main__':
    t = main()
    t.mainFunc()
