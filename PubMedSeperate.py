import sys
import random

def Seperate(rawFile, trainRate, testRate):
    trainWrite = open("2train-"+rawFile, 'w')
    testWrite = open("2test-"+rawFile, 'w')
    random.seed(0)
    for line in open(rawFile):
        x = random.uniform(0, 1)
        if x < trainRate:
            trainWrite.write(line)
        else:
            testWrite.write(line)
    trainWrite.close()
    testWrite.close()

if __name__=="__main__":
    rawFile = sys.argv[1]
    trainRate = 0.994
    testRate = 0.006
    Seperate(rawFile, trainRate, testRate)
