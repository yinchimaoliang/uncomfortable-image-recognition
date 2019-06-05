import matplotlib.pyplot as plt

LOG_PATH = './log'



class main():
    def __init__(self):
        self.test_file = LOG_PATH + '/' + 'test_res34.txt'
        self.train_file = LOG_PATH + '/' + 'train_res34.txt'


        self.train_log = []
        self.test_log = []


    def getData(self):
        with open(self.train_file,'r') as f:
            lines = f.readlines()
            for line in lines:
                epoch,rate = line.split(':')
                self.train_log.append(float(rate[:-2]))

        with open(self.test_file,'r') as f:
            lines = f.readlines()
            for line in lines:
                epoch,rate = line.split(':')
                self.test_log.append(float(rate[:-2]))



    def drawData(self):
        x_label = [i + 1 for i in range(len(self.test_log))]
        plt.axis([0,30,0,100])
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.plot(x_label,self.train_log,'r',label = 'train')
        plt.plot(x_label,self.test_log,'g',label = 'test')
        plt.legend()
        plt.show()






    def mainFunc(self):
        self.getData()
        self.drawData()



if __name__ == '__main__':
    t = main()
    t.mainFunc()
