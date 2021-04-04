import time
import string
import math
import numpy as np
import pandas as pd
import random
from collections import defaultdict, Counter
from statistics import mean,stdev
from sklearn.svm import SVC

def load_data(dataset,datatype):
    # load digit and face data according to given datatype
    # load train and test data according to given dataset
    if (datatype=='d'):
        if(dataset=="Train"):
            imagefile= open("C:/Rutgers/Courses/Fall_2020/Intro_to_AI/Final/data/digitdata/trainingimages","r")
            labelfile = open("C:/Rutgers/Courses/Fall_2020/Intro_to_AI/Final/data/digitdata/traininglabels",'r')
        else:
            imagefile = open("C:/Rutgers/Courses/Fall_2020/Intro_to_AI/Final/data/digitdata/testimages", "r")
            labelfile = open("C:/Rutgers/Courses/Fall_2020/Intro_to_AI/Final/data/digitdata/testlabels",'r')
    else:
        if (dataset == "Train"):
            imagefile = open("C:/Rutgers/Courses/Fall_2020/Intro_to_AI/Final/data/facedata/facedatatrain", "r")
            labelfile = open("C:/Rutgers/Courses/Fall_2020/Intro_to_AI/Final/data/facedata/facedatatrainlabels", 'r')
        else:
            imagefile = open("C:/Rutgers/Courses/Fall_2020/Intro_to_AI/Final/data/facedata/facedatatest", "r")
            labelfile = open("C:/Rutgers/Courses/Fall_2020/Intro_to_AI/Final/data/facedata/facedatatestlabels", 'r')
    lines= imagefile.read().splitlines()
    images= []
    count = 0
    temp = []
    if(datatype=='d'):
        pixels=28
    else:
        pixels=70

    for line in lines:
        count += 1
        temp.append(line)
        if(count == pixels):
            images.append(temp)
            count = 0
            temp = []

    lines = labelfile.read().splitlines()
    labels = []
    for line in lines:
        labels.append(int(line))
    return images, labels

def trainPerceptron(images, labels, trainingSize,datatype,test):
    start = time.time()
    if (test == "classification"):
        last = int((float(trainingSize / 100.0)) * len(images))
    else:
        last = len(images)

    wchange = True
    if(datatype=='d'):
        weights = []
        i = 0
        while (i < 10):
            tempArray = [0 for count in range(len(images[0]) * len(images[0][0]))]
            weights.append(tempArray)
            i += 1
        biasSet = [0] * 10
    else:
        weights = [0 for count in range(len(images[0]) * len(images[0][0]))]
        biasSet = 0

    while wchange:
        wchange = False
        for image in images[0:last]:
            if(datatype=='d'):
                vals = [0.0]*10
                j = 0
                while (j < 10):
                    vals[j] = Activationfunction(image, weights[j], biasSet[j])
                    j += 1
                if (vals.index(max(vals)) != labels[images.index(image)]):
                    # Update weights
                    weights[vals.index(max(vals))], biasSet[vals.index(max(vals))]=Weightchange(image, weights[vals.index(max(vals))], biasSet[vals.index(max(vals))], -1)
                    weights[int(labels[images.index(image)])], biasSet[int(labels[images.index(image)])]=Weightchange(image, weights[int(labels[images.index(image)])], biasSet[int(labels[images.index(image)])], 1)
            else:
                vals = Activationfunction(image, weights, biasSet)
                if ((vals >= 0) and (labels[images.index(image)] == 0)):
                    weights, biasSet = Weightchange(image, weights, biasSet, -1)
                    wchange = True
                elif (vals < 0 and (labels[images.index(image)]) == 1):
                    weights, biasSet = Weightchange(image, weights, biasSet, 1)
                    wchange = True
    end = time.time()
    runtime = end - start
    return weights, biasSet, runtime

def testPerceptron(images,labels, weights, bias, trainingSize, datatype, runtime):
    correct = 0
    incorrect = 0
    for image in images:
        if(datatype=='d'):
            vals = [0.0]*10
            j = 0
            while (j < 10):
                vals[j] = Activationfunction(image, weights[j], bias[j])
                j += 1
            if (vals.index(max(vals)) != labels[images.index(image)]):
                incorrect += 1
            else:
                correct += 1
        else:
            vals = Activationfunction(image, weights, bias)
            if ((vals >= 0 and labels[images.index(image)] == 1) or (vals < 0 and labels[images.index(image)] == 0)):
                correct += 1
            else:
                incorrect += 1
    percentCorrect = float(correct / float(correct + incorrect)) * 100
    percentIncorrect = float(incorrect / float(correct + incorrect)) * 100
    print("Training Set Size: " + str(trainingSize) + "%")
    print("Runtime: " + str(runtime))
    print("Correct: " + str(percentCorrect) + "%")
    print("Incorrect: " + str(percentIncorrect) + "%")

    return percentCorrect

def trainNaive(images, labels, trainingSize,datatype,test):
    start = time.time()
    # Amount of training data to be used
    if(test=="classification"):
        last = int((float(trainingSize / 100.0)) * len(images))
    else:
        last = len(images)
    # Calculate priors for digits 0-9
    priors=[]
    imageTables = []
    if(datatype=='d'):
        for i in range(0,10):
            priors.append(float(labels[0:last].count(i))/float(last))
            table = [0.0 for count in range(len(images[0]) * len(images[0][0]))]
            imageTables.append(table)

    else:
        facepriors=float(labels.count(1))/float(len(labels))
        Notfacepriors=float(labels.count(0))/float(len(labels))
        priors.append(facepriors)
        priors.append(Notfacepriors)
        table = [0.0 for count in range(len(images[0]) * len(images[0][0]))]
        imtable=[0.0 for count in range(len(images[0]) * len(images[0][0]))]
        imageTables.append(table)
        imageTables.append(imtable)

    # Construct an array to load image data
    # Load data other than empty
    for image in images[0:last]:
        if(datatype=='d'):
            currentimageind = int(labels[images.index(image)])
        else:
            if(labels[images.index(image)] == 1):
                currentimageind=0
            else:
                currentimageind=1

        k = 0
        for i in image:
            for j in i:
                if (j != ' '):
                    imageTables[currentimageind][k] += 1.0
                k += 1

    for im in range(len(imageTables)):
        for jm in range(len(imageTables[im])):
            if imageTables[im][jm] > 0.0:
                if(datatype=='f'):
                    if(im==0):
                        imageTables[im][jm] = float(imageTables[im][jm]) / float(labels[0:last].count(1))
                    else:
                        imageTables[im][jm] = float(imageTables[im][jm]) / float(labels[0:last].count(0))
                else:
                    imageTables[im][jm] = float(imageTables[im][jm]) / float(labels[0:last].count(im))
            else:
                imageTables[im][jm] = 0.0001
    end = time.time()
    runtime = end - start
    return imageTables, priors, runtime

def testDigitNaive(images, labels, tables, priors, trainingSize, runtime):
    correct = 0
    incorrect = 0
    for image in images:
        pDigits, decimalShifts = DigitProbability(image, tables, priors)
        prediction = decimalShifts.index(min(decimalShifts))
        sameshift = []
        count = 0
        #Check if decimalshifts is same for any other digit
        for i in range(len(decimalShifts)):
            if decimalShifts[i] == decimalShifts[prediction]:
                count += 1
                sameshift.append(i)
        flag = -1
        #If the decimal shift is same, then check for prior absolute value
        if count > 1:
            # Get the max of pDigits out of the indexes in duplicates
            tempMax = 0
            for j in range(len(pDigits)):
                if j in sameshift:
                    if pDigits[j] > tempMax:
                        tempMax = pDigits[j]
                        flag = 1
            if (flag == 1):
                prediction = tempMax

        if labels[images.index(image)] == prediction:
            correct += 1
        else:
            incorrect += 1

    percentCorrect = float(correct / float(correct + incorrect)) * 100
    percentIncorrect = float(incorrect / float(correct + incorrect)) * 100

    print("Training Set Size: " + str(trainingSize) + "%")
    print("Runtime: " + str(runtime))
    print("Correct: " + str(percentCorrect) + "%")
    print("Incorrect: " + str(percentIncorrect) + "%")

    return percentCorrect

def testFaceNaive(images, labels, FaceTable, NotFaceTable, priorFace, priorNotFace, trainingSize,runtime):
    correct = 0
    incorrect = 0
    for image in images:
        pFace, decimalShift1 = ImageProbability(image, FaceTable, priorFace)
        pNotFace, decimalShift2 = ImageProbability(image, NotFaceTable, priorNotFace)
        difference = decimalShift1 - decimalShift2

        if ((difference == 0 and pFace >= pNotFace) or difference < 0):
            if (labels[images.index(image)] == 0):
                incorrect += 1
            else:
                correct += 1
        elif ((difference == 0 and pFace < pNotFace) or difference > 0):
            if (labels[images.index(image)] == 1):
                incorrect += 1
            else:
                correct += 1
    percentCorrect = float(correct / float(correct + incorrect)) * 100
    percentIncorrect = float(incorrect / float(correct + incorrect)) * 100
    print("Training Set Size: " + str(trainingSize) + "%")
    print("Runtime: " + str(runtime))
    print("Correct: " + str(percentCorrect) + "%")
    print("Incorrect: " + str(percentIncorrect) + "%")

    return percentCorrect

def DigitProbability(image, tables, priors):
    vals = [1] * 10
    k = 0
    decimalShifts = [0] * 10
    for j in image:
        for i in j:
            for x in range(len(vals)):
                if (i != ' '):
                    vals[x] = vals[x] * tables[x][k]
                else:
                    vals[x] = vals[x] * (1-tables[x][k])


                if (vals[x] < 0.1):
                    vals[x] = vals[x] * 10
                    decimalShifts[x] += 1
            k += 1
    for n in range(len(vals)):
        vals[n] = vals[n] * priors[n]

    return vals, decimalShifts

def Weightchange(image, weights, bias, change):
    k = 0
    if (change > 0):  # Increase weights
        bias += 1
    else:
        bias -= 1
    for i in image:
        for j in i:
            if (j != ' '):
                if(change>0):
                    weights[k] += 1
                else:
                    weights[k] -= 1
            k += 1
    return weights, bias

def Activationfunction(image, weights, bias):
    fValue = 0
    fValue += bias
    k = 0;
    for i in image:
        for j in i:
            if (j == ' '):
                fValue += 0
            else:
                fValue += weights[k]
            k += 1
    return fValue

def ImageProbability(image, featureTable, prior):
    val = 1
    k = 0
    decimalShift = 0
    for j in image:
        for i in j:
            if (i != ' '):
                val = val * featureTable[k]
            else:
                val = val * (1-featureTable[k])
            k += 1
            if (val < 0.1):
                val = val * 10
                decimalShift += 1

    return (val * prior), decimalShift

def euclidean_distance(testimage, trainimage):
    return np.sqrt(sum((testimage - trainimage) ** 2))

def Getprediction(labels):
    occurence_count = Counter(labels)
    prediction= occurence_count.most_common(1)[0][0]
    return prediction

def predict(k, train_images, train_labels, test_image):
    # distances contains tuples of (euclidean_distance, label)
    # euclidean_distance(test_image, image)
    distances = [[euclidean_distance(test_image, image), label] for (image, label) in zip(train_images, train_labels)]
    # sort the distances list by distance
    distances.sort(key=lambda r: r[0])
    # extract only k closest labels
    k_labels = [label for (_, label) in distances[:k]]
    return Getprediction(k_labels)

def testKNN(images,labels,testimages,testlabels,trainingSize):
    i = 0
    correct = 0
    incorrect = 0
    start = time.time()
    last = int((float(trainingSize / 100.0)) * len(images))
    for image in testimages:
        pred = predict(6, images[0:last], labels[0:last], image)
        if pred == testlabels[i]:
            correct += 1
        else:
            incorrect += 1
        i += 1
    end = time.time()
    runtime = end - start
    percentCorrect = float(correct) / float(correct + incorrect) * 100
    percentIncorrect = float(incorrect) / float(correct + incorrect) * 100
    print("Face Training Set Size: " + str(trainingSize) + "%")
    print("Runtime: " + str(runtime))
    print("Correct: " + str(percentCorrect) + "%")
    print("Incorrect: " + str(percentIncorrect) + "%")

def img_vec(vec):
    b = [list(i) for i in vec]
    char_to_replace = {' ': 0, '#': 1, '+': 1}
    return [char_to_replace.get(n, n) for i in b for n in i]

def AccuracyPrediction(Images,Labels,TestImages,TestLabels,size,datatype,test):
    last = int((float(size / 100.0)) * len(Images))
    randomlist = random.sample(range(0, len(Images)), last)
    RandomImages=[]
    RandomLabels=[]
    for val in randomlist:
        RandomImages.append(Images[val])
        RandomLabels.append(Labels[val])

    tables,priors,runtime=trainNaive(RandomImages,RandomLabels,size,datatype,test)
    if(datatype=='d'):
        accuracyNaive= testDigitNaive(TestImages,TestLabels,tables,priors,size,runtime)
    else:
        accuracyNaive = testFaceNaive(TestImages,TestLabels,tables[0],tables[1],priors[0],priors[1],size,runtime)

    weights, bias, runtime = trainPerceptron(RandomImages, RandomLabels, size, datatype,test)
    accuracyPercep= testPerceptron(TestImages, TestLabels, weights, bias, size, datatype, runtime)

    return accuracyNaive, accuracyPercep


if __name__ == "__main__":

    while True:
        datatype= input("Enter f for Faces or d for Digits.\n").lower()
        if (datatype != 'f' and datatype != 'd'):
            print("Improper input Try again.\n")
        else:
            break

    while True:
        method = input("Enter p for Perceptron, n for Naive Bayes, or k for KNN Classifiers.\n").lower()
        if (method != 'p' and method != 'n' and method != 'k'):
            print("Improper input Try again.\n")
        else:
            break

    while True:
        size = int(input("Enter the percentage of training set images to be used (must be multiple of 10).\n"))              # Possible 10,20,30,40,50,60,70,80,90,100
        if ((size % 10) != 0 or size > 100):
            print("Improper input Try again.\n")
        else:
            break

    test="classification"
    #test = "accuracy"

    if(test=="classification"):
        if (datatype == 'd'):
            Image, Labels = load_data("Train", datatype)
            TestImage, TestLabel = load_data("Test", datatype)
        else:
            Image, Labels = load_data("Train", datatype)
            TestImage, TestLabel = load_data("Test", datatype)

        if(method=="n"):
            table, priors, runtime = trainNaive(Image,Labels, size,datatype,test)
            if(datatype=='d'):
                testDigitNaive(TestImage, TestLabel, table, priors, size, runtime)
            else:
                testFaceNaive(TestImage,TestLabel,table[0],table[1],priors[0],priors[1],size,runtime)
        elif(method=="p"):
            weights, bias, runtime = trainPerceptron(Image, Labels, size,datatype,test)
            testPerceptron(TestImage, TestLabel, weights, bias, size,datatype, runtime)
        else:
                temp1 = map(img_vec, Image)
                k_dImages = list(temp1)
                temp2 = map(img_vec, TestImage)
                k_dTestImages = list(temp2)

                train_images = np.asarray(k_dImages)
                train_labels = np.asarray(Labels)
                test_images = np.asarray(k_dTestImages)
                test_labels = np.asarray(TestLabel)

                testKNN(train_images,train_labels, test_images,test_labels,size)
    else:
        digitnacc = []
        facenacc = []
        digitpacc = []
        facepacc = []
        dImage, dLabels = load_data("Train", "d")
        dTestImage, dTestLabel = load_data("Test", "d")
        fImage, fLabels = load_data("Train", "f")
        fTestImage, fTestLabel = load_data("Test", "f")

        for si in range(10,110,10):
            for i in range(5):
                dnaive,dpercep=AccuracyPrediction(dImage,dLabels,dTestImage,dTestLabel,si,"d",test)
                fnaive,fpercep=AccuracyPrediction(fImage, fLabels, fTestImage, fTestLabel, si, "f",test)
                digitnacc.append(dnaive)
                facenacc.append(fnaive)
                digitpacc.append(dpercep)
                facepacc.append(fpercep)

            print("For Digit Classification and Training Set Size: " + str(si) + "%")
            print("Mean: " + str(mean(digitnacc)) + "%")
            print("Standard Deviation: " + str(stdev(digitnacc)) + "%")

            print("For Face Classification and Training Set Size: " + str(si) + "%")
            print("Mean: " + str(mean(facenacc)) + "%")
            print("Standard Deviation: " + str(stdev(facenacc)) + "%")

            print("For Digit Classification and Training Set Size: " + str(si) + "%")
            print("Mean: " + str(mean(digitpacc)) + "%")
            print("Standard Deviation: " + str(stdev(digitpacc)) + "%")

            print("For Face Classification and Training Set Size: " + str(si) + "%")
            print("Mean: " + str(mean(facepacc)) + "%")
            print("Standard Deviation: " + str(stdev(facepacc)) + "%")