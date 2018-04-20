import sys
import random

def Seperate(rawFile, trainRate, testRate):
    trainWrite = open("train-validate-"+rawFile, 'w')
    testWrite = open("test-"+rawFile, 'w')
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
    trainRate = 0.95
    testRate = 0.05
    Seperate(rawFile, trainRate, testRate)
