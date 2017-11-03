#KNN implemenation by Mayuri Wadkar

import sys
import math
import re
import operator
import matplotlib.pyplot as plt
import numpy as np

class Agent:
    def calculateEucledeanDistance(self,percept,trainDatasetRecord):
        #this method calculates float Euclidean distance between a percept from test dataset and a record from training dataset
        d = 0.0
        for i in range(0,len(trainDatasetRecord)-1):
            x = float(percept[i])
            y = float(trainDatasetRecord[i])
            d += ((x-y) ** 2)
        d = math.sqrt(d)
        return d

    def classPrediction(self,percept,handle_trainDataset,K):
        #this method returns a predicted class attribute for percept received from Environment
        eu_Distance = []
        knn = []

        handle_trainDataset.seek(0)
        line = None

        while line != '':
            line = handle_trainDataset.readline()
            if line.strip() == "@data":
                break

        while line != '':
            line = handle_trainDataset.readline().strip()
            if line != "":
                trainDatasetRecord = line.split(", ")
            else:
                break
            #Calculating euclidean distance between received percept from test dataset fold with current record in train data set
            d = self.calculateEucledeanDistance(percept,trainDatasetRecord)
            eu_Distance.append((trainDatasetRecord[-1],d))

        eu_Distance.sort(key = operator.itemgetter(1))

        #List of K nearest neighbours of the precept
        knn = eu_Distance[:int(K)]
        class1 = 0
        class2 = 0
        for k in knn:
            if k[0] == Environment.classLabels[0]:
                class1 += 1
            else:
                class2 += 1

        #maximum frequency class label is assigned to percept
        if class1 > class2:
            return Environment.classLabels[0]
        else:
            return Environment.classLabels[1]

class Environment:
    testSetFileHandle = []
    trainSetFileHandle = []
    classLabels = []
    TestDatasetAccuracies = []
    AvgAccuracyForK = []

    def __init__(self):
        pass

    def loadDatasets(self,inputDir,datasetName):
        # this method loads training & test dataset handles into an array of handles

        for trainSetId in range(10):
            filename = str(inputDir + "/" + datasetName + "-10-" + str(trainSetId+1) + "tra.dat")
            handle1 = open(filename,'r')
            self.trainSetFileHandle.append(handle1);

        for testSetId in range(10):
            filename = str(inputDir + "/" + datasetName + "-10-" + str(testSetId+1) + "tst.dat")
            handle2 = open(filename,'r')
            self.testSetFileHandle.append(handle2);
        handle1.seek(0)
        line = None
        classLabel = ""
        while line != '':
            line = handle1.readline()
            if line.split(" ")[0] == "@outputs" or line.split(" ")[0] == "@output":
                classLabel = line.strip().split(" ")[1]
                break
        handle1.seek(0)

        #this code snippet fetches binary class label values from datasets
        line = None
        while line != "":
            line = handle1.readline().strip()
            list_token = re.split(r'[ },;{]+', line)
            if list_token[0] == "@attribute":
                if list_token[1] == classLabel:
                    i = line.find("{")
                    self.classLabels.append(list_token[2])
                    self.classLabels.append(list_token[3])
                    break
        handle1.seek(0)

    def closeDatasets(self):
        # this method closes all training dataset file handles
        for handle in self.testSetFileHandle:
            handle.close()
        for handle in self.trainSetFileHandle:
            handle.close()

    def sendPercepts(self,handle_testDt,handle_trainDt,agent,K):
        #this method sends percepts to agent for finding k nearest neighbours
        handle_testDt.seek(0)
        line = None

        # read test dataset file until you hit @data
        while line != '':
            line = handle_testDt.readline()
            if line.strip() == "@data":
                break

        # read all the precepts for this fold
        perceptList = []
        while line != '':
            line = handle_testDt.readline().strip()
            if line != "":
                percept = line.split(", ")
                perceptList.append(percept)

        # sending percepts to Agent
        predictedClassLable = []
        for percept in perceptList:
            classLable = agent.classPrediction(percept,handle_trainDt,K)
            predictedClassLable.append(classLable)

        accuracy = float(self.findAccuracy(perceptList, predictedClassLable))
        # storing accuracy for every round to find average accuracy
        self.TestDatasetAccuracies.append(float(accuracy))

    def findAccuracy(self,perceptList,predictedClassLable):
        #compares predicted class for a percept with corresponding class labels from training records
        #finds average accuracy over one test data fold
        correct = 0
        for i in range(0,len(predictedClassLable)):
            if (perceptList[i][-1]) == (predictedClassLable[i]):
                correct += 1
        accuracyPercentage = (float(correct)/len(predictedClassLable))*100

        print "Accuracy = ",
        print float(accuracyPercentage),
        print "%\t"
        return accuracyPercentage

    def findAvgAccuracyForK(self):
        #finding average accuracy of a classifier for K ranging from 1 to 10
        avg = 0.0
        for a in Environment.TestDatasetAccuracies:
            avg += float(a)
        avg = float(avg)/len(Environment.TestDatasetAccuracies)
        return avg

def plotGraph(x,y):
    #ploting graph of K vs average accuracy
    x_axis = np.array(x)
    y_axis = np.array(y)
    plt.plot(x_axis,y_axis)
    plt.xlabel('X-axis  -  K')
    plt.ylabel('Y-axis  -  Average Accuracy (%)')
    title = str(sys.argv[2])+' Dataset : K vs Average accuracy'
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    #taking inputs from command line for input directory, dataset name and value of K
    datasetDirectory = sys.argv[1]
    dataset = sys.argv[2]
    K = sys.argv[3]
    xAxis = []
    yAxis = []

    env = Environment()
    agent = Agent()

    env.loadDatasets(datasetDirectory,dataset)

    print ">>>Dataset:"+str(dataset)
    print "For K = "+str(K)+" -->"
    for j in range(10):
        handle_trainDt = env.trainSetFileHandle[j]
        handle_testDt = env.testSetFileHandle[j]
        print "For Test Dataset " + str(j + 1) + "\t",
        env.sendPercepts(handle_testDt, handle_trainDt, agent, K)
    avg_accuracy = env.findAvgAccuracyForK()
    print "\nAverage Accuracy of a classifier for K = " + str(K) + ":\t" + str(avg_accuracy) + "\n"

    print "Analyzing Average Accuracy values for different values of K(1-10)......"
    for k_value in range(1,11):
        Environment.TestDatasetAccuracies = []
        xAxis.append(k_value)
        print "For K = "+str(k_value)+" -->"
        for j in range(10):
            handle_trainDt = env.trainSetFileHandle[j]
            handle_testDt = env.testSetFileHandle[j]
            print "For Test Dataset " + str(j + 1) + "\t",
            env.sendPercepts(handle_testDt, handle_trainDt, agent, k_value)
        avg_accuracy = env.findAvgAccuracyForK()
        print "\nAverage Accuracy of a classifier for K = "+str(k_value)+":\t"+str(avg_accuracy)+"\n"
        yAxis.append(avg_accuracy)

    plotGraph(xAxis,yAxis)
    env.closeDatasets()

if __name__ == '__main__':
        main()