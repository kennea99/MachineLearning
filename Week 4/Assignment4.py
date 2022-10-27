import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import Ridge
import random
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsRegressor

def readFiles():
    out1 = None
    out2 = None
    with open("week4.txt") as contained:
        for line in contained:
            line = line.strip()
            if line.startswith('# id') and out1 is None:
                out1 = open("week4_first.txt", 'w')
                out1.write(line +'\n')
            elif line.startswith('# id') and out2 is None:
                out2 = open('week4_second.txt', 'w')
                out2.write(line + '\n')
            
            elif out1 is not None and out2 is None:
                out1.write(line + '\n')
            
            elif out2 is not None:
                out2.write(line+'\n')

def splitData(filename):
    df = pd.read_csv(filename, comment = '#')
    X1 = df.iloc[:,0]
    X2 = df.iloc[:,1]
    targetVal = df.iloc[:,2]
    X = np.column_stack((X1,X2))
    return X1, X2, targetVal, X

def selectClassifier(fit, regModel, cVal, X, targetVal):
    if regModel == "Logistic":
        model = LogisticRegression(penalty='l2', solver='lbfgs', C = cVal, max_iter=100000)
    model = model.fit(fit, targetVal)
    return model

def getPoly(degree, X, targetVal):
    poly = PolynomialFeatures(degree)
    fit = poly.fit_transform(X)
    return fit

def kCrossValid(X, targetVal, cVal, k, regModel, degree):
    model = selectClassifier(X, "Logistic", cVal, X, targetVal)
    polyFit = getPoly(degree, X, targetVal)
    scores = cross_val_score(model, polyFit, targetVal, cv = 10, scoring = 'f1')
    return scores.mean(), scores.var(), scores.std()

def kCrossValidPoly(model, X, targetVal, polyValue, k):
    scores = cross_val_score(model, X, targetVal, cv=10, scoring='f1')
    return scores.mean(), scores.var(), scores.std()

def fold10_cVals(X, targetVal, cVals, regModel, degree):
    meanArray = np.array([0]*len(cVals), dtype=float)
    stdArray = np.array([0]*len(cVals), dtype=float)
    i = 0
    for i in range(len(cVals)):
        mean, var, std = kCrossValid(X, targetVal, cVals[i], 10, "Logistic", degree)
        meanArray[i] = mean
        stdArray[i] = std     
    return meanArray, stdArray

def fold10_polyFeat(model, X, targetVal, polyValue):
    mean, var, std = kCrossValidPoly(model, X, targetVal, polyValue, 10)
    return mean, std

def getBestK(kVals, XTrain, TargetValTrain):
    meanArray = np.array([0] * len(kVals), dtype= float)
    stdArray = np.array([0] * len(kVals), dtype=float)
    for i in range(len(kVals)):
        model = KNeighborsClassifier(n_neighbors = kVals[i], weights = 'uniform').fit(XTrain, TargetValTrain)
        scores = cross_val_score(model, XTrain, TargetValTrain, cv = 10, scoring='f1')
        meanArray[i] = scores.mean()
        stdArray[i] = scores.std()
    return meanArray, stdArray

def getBestKAugmentPoly(kVals, XTrain, TargetValTrain,n):
    meanArray = np.array([0] * len(kVals), dtype= float)
    stdArray = np.array([0] * len(kVals), dtype=float)
    poly = PolynomialFeatures(n)
    polyFit = poly.fit_transform(XTrain)
    for i in range(len(kVals)):
        model = KNeighborsClassifier(n_neighbors = kVals[i], weights = 'uniform')#.fit(polyFit, TargetValTrain)
        scores = cross_val_score(model, polyFit, TargetValTrain, cv = 10, scoring='f1')
        meanArray[i] = scores.mean()
        stdArray[i] = scores.std()
    return meanArray, stdArray

def plot_error(cVals, mean, std, colors, lineColor, title, xLabel, yLabel):
    mpl.errorbar(cVals, mean, yerr=std, ecolor=colors, color = lineColor)
    mpl.title(title)
    mpl.xlabel(xLabel)
    mpl.ylabel(yLabel)

def first_plot(X1, X2, targetVal, title):
    pointColours = ['red'] * len(targetVal)
    for i in range(len(pointColours)):
        if targetVal[i] == -1:
            pointColours[i] = 'cornflowerblue'
    for labels in ["target value y = 1", "target value y = -1"]:
        mpl.scatter(X1, X2, c = pointColours, marker = '+', label = labels)
    mpl.title(title)
    mpl.xlabel("feature X1")
    mpl.ylabel("feature X2")
    mpl.legend()
    mpl.gca()
    legend = mpl.legend()
    legend.legendHandles[0].set_color('red')
    mpl.show()

def second_plot(X1, X2, targetVal, title):
    pointColours = ['red'] * len(targetVal)
    for i in range(len(pointColours)):
        if targetVal[i] == -1:
            pointColours[i] = 'cornflowerblue'
    for labels in ["target value y = 1", "target value y = -1"]:
        mpl.scatter(X1, X2, c = pointColours, marker = '+', label = labels)
    mpl.title(title)
    mpl.xlabel("feature X1")
    mpl.ylabel("feature X2")
    mpl.legend()
    mpl.gca()
    legend = mpl.legend()
    legend.legendHandles[1].set_color('cornflowerblue')
    mpl.show()

def classifier(yPredict, color1, color2):
    predictColours = [color1] * len(yPredict)
    for i in range(len(yPredict)):
        if(yPredict[i] == -1):
            predictColours[i] = color2
    return predictColours


def main():
    readFiles()
    X1, X2, firstTargetVal, firstX = splitData('week4_first.txt')
    X3, X4, secondTargetVal, secondX = splitData('week4_second.txt')
    first_plot(X1, X2, firstTargetVal, "ScatterPlot of Dataset 1")
    cVals = np.array([0.01, 0.1, 1, 2, 5, 10, 20, 30])
    degrees = np.array([2,4,6,8,10,12,14])
#USING THE FIRST DATASET HERE, WILL SPECIFY WHEN USING THE SECOND DATASET
    meanArrayC, stdArrayC = fold10_cVals(firstX, firstTargetVal, cVals, 'Logistic', 6)
    polyMeanArray = np.array([0]*len(degrees), dtype=float)
    polyStdArray = np.array([0]*len(degrees), dtype=float)
    for n in range(len(degrees)):
        fit = getPoly(degrees[n], firstX, firstTargetVal)
        model = selectClassifier(fit, "Logistic", 1, firstX, firstTargetVal)
        mean, std = fold10_polyFeat(model, fit, firstTargetVal, degrees[n])
        polyMeanArray[n] = mean
        polyStdArray[n] = std
    print(polyStdArray)
    plot_error(cVals, meanArrayC, stdArrayC,"grey","blue", "10-fold cross validation, mean and  standard deviation of f1 scoring vs C", "Values of C using Logistic", "F1 scoring mean")
    leg_point0 = mpl.Rectangle((0,0), 1, 1, fc = 'grey')
    leg_point1 = mpl.Rectangle((0,0), 1,1 , fc = 'blue')
    mpl.legend([leg_point0,leg_point1], ["f1 score mean", "Standard Deviation"])
    mpl.show()

    plot_error(degrees, polyMeanArray, polyStdArray,"grey","blue", "10-fold cross validation, f1 scoring vs degrees of Polynomial Features", "PolynomialFeatures(n)", "F1 scoring mean")
    leg_point0 = mpl.Rectangle((0,0), 1, 1, fc = 'grey')
    leg_point1 = mpl.Rectangle((0,0), 1,1 , fc = 'blue')
    mpl.legend([leg_point0,leg_point1], ["Standard Deviation", "f1 scoring mean"])
    mpl.show()

#Logistic Regression Classifier using best polyfeature and C Value.
    bestPolyFit = getPoly(6, firstX, firstTargetVal)
    model = LogisticRegression(penalty='l2', solver='lbfgs', C = 1)
    model.fit(bestPolyFit, firstTargetVal)
    ypred = model.predict(bestPolyFit)
    ax = mpl.subplot()
    trainColor = classifier(firstTargetVal, 'red', 'cornflowerblue')
    predictColour = classifier(ypred, 'black', 'lime')
    for labels in ["Trained data y = 1", "Trained data y = -1"]:
        ax.scatter(X1, X2, c = trainColor, label = labels)
    for labels in ["Predicted data y = 1", "Predicted data y = -1"]:
        ax.scatter(X1, X2, c = predictColour, marker = '+', label = labels, s= 5**2)
    mpl.xlabel("feature X1")
    mpl.ylabel("feature X2")
    mpl.title("Logistic Regression Classifier with polyfeature of degree 6 and C = 1")
    mpl.legend(loc = 'lower center')
    ax = mpl.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('red')
    leg.legendHandles[2].set_color('black')
    leg.legendHandles[3].set_color('lime')
    mpl.show()


    #nearest neighbors
    firstXTrain, firstXTest, firstTargetValTrain, firstTargetValTest = train_test_split(firstX, firstTargetVal, test_size=0.2)
    kVals = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    meanArrayK, stdArrayK = getBestK(kVals, firstXTrain, firstTargetValTrain)
    print(stdArrayK)
    meanPolyK, stdPolyK = getBestKAugmentPoly(kVals, firstXTrain, firstTargetValTrain, 2)
    meanPolyK1, stdPolyK1 = getBestKAugmentPoly(kVals, firstXTrain, firstTargetValTrain, 3)

    plot_error(kVals, meanArrayK, stdArrayK, "lime", 'red', "10-fold cross validation of kNN to determine best k", "k", "f1 scoring mean" )
    leg_point0 = mpl.Rectangle((0,0), 1, 1, fc = 'lime')
    leg_point1 = mpl.Rectangle((0,0), 1,1 , fc = 'red')
    mpl.legend([leg_point0,leg_point1], ["Standard Deviation", "f1 scoring mean"])
    mpl.show()
    plot_error(kVals, meanArrayK, stdArrayK, "lime", 'red', None, None, None )
    plot_error(kVals, meanPolyK1, stdPolyK1, "pink", 'green', None, None, None )
    plot_error(kVals, meanPolyK, stdPolyK, "grey", 'blue', None, None, None )
    mpl.title("kNN using no augmented and augmented polynomial features")
    mpl.xlabel("k = number of neighbors")
    mpl.ylabel("f1 scoring mean")
    mpl.legend(["no Polyfeatures", "PolyFeatures(2)", "PolyFeatures(3)"], loc='lower right')
    mpl.show()

    #using best value of k
    model = KNeighborsClassifier(n_neighbors=8, weights = 'uniform').fit(firstXTrain, firstTargetValTrain)
    firstYPred = model.predict(firstXTest)
    firstTargetValTest = np.asarray(firstTargetValTest)
    firstTargetValTrain = np.asarray(firstTargetValTrain)
    testColor = classifier(firstTargetValTest, 'red', 'cornflowerblue')
    predictColour = classifier(firstYPred, 'black', 'lime')
    for labels in ["Test data y = 1", "Test data y = -1"]:
        mpl.scatter(firstXTest[:,0], firstXTest[:,1], c = testColor, label = labels)
    for labels in ["Predicted Test data y = 1", "Predicted Test data y = -1"]:
        mpl.scatter(firstXTest[:,0], firstXTest[:,1], c = predictColour, marker = '+', label = labels)
    mpl.xlabel("feature X1")
    mpl.ylabel("feature X2")
    mpl.title("kNN Classifier k = 8 using Testing Data")
    mpl.gca()
    leg = mpl.legend()
    leg.legendHandles[0].set_color('red')
    leg.legendHandles[1].set_color('cornflowerblue')
    leg.legendHandles[2].set_color('black')
    leg.legendHandles[3].set_color('lime')
    mpl.show()

    #CONFUSTION MATRICES first dataset
    dummyModel = DummyClassifier(strategy="most_frequent").fit(firstXTrain, firstTargetValTrain)
    dummyPred = dummyModel.predict(firstXTest)

    print("Confusion Matrix for Logistic Regression Classifier")
    print(confusion_matrix(firstTargetVal, ypred))

    print("Confusion Matrix for kNN")
    print(confusion_matrix(firstTargetValTest, firstYPred))

    print("Confusion Matrix for Dummy Classifier")
    print(confusion_matrix(firstTargetValTest, dummyPred))

    #ROC Curves
#Logistic
    Xtrain, Xtest, ytrain, ytest = train_test_split(bestPolyFit, firstTargetVal, test_size=0.2)
    logisticModel = LogisticRegression(penalty='l2', solver='lbfgs', C = 1)
    logisticModel.fit(Xtrain, ytrain)
    fprLog, tprLog,_ = roc_curve(ytest, logisticModel.decision_function(Xtest))
    mpl.plot(fprLog, tprLog)
#kNN
    y_scores = model.predict_proba(firstXTest)
    fprknn, tprknn, _ = roc_curve(firstTargetValTest, y_scores[:,1])
    mpl.plot(fprknn, tprknn)
#baseline
    mpl.plot([0, 1], [0, 1], color='green',linestyle='--')
    mpl.xlabel("False Positive Rate")
    mpl.ylabel("True Positive Rate")
    mpl.title("ROC Curve with Logistic Regression and knn and baseline classifier")
    mpl.legend(["Logistic", "kNN", "Baseline"])
    mpl.show()

#USING THE SECOND DATASET NOW!!!

    second_plot(X3, X4, secondTargetVal, "ScatterPlot of Dataset 2")
    secondMeanArrayC, secondStdArrayC = fold10_cVals(secondX, secondTargetVal, cVals, 'Logistic', 2)
    secondPolyMeanArray = np.array([0]*len(degrees), dtype=float)
    secondPolyStdArray = np.array([0]*len(degrees), dtype=float)
    for n in range(len(degrees)):
        fit = getPoly(degrees[n], secondX, secondTargetVal)
        model = selectClassifier(fit, "Logistic", 1, secondX, secondTargetVal)
        mean, std = fold10_polyFeat(model, fit, secondTargetVal, degrees[n])
        secondPolyMeanArray[n] = mean
        secondPolyStdArray[n] = std

    plot_error(cVals, secondMeanArrayC, secondStdArrayC,"grey","blue", "10-fold cross validation, mean and  standard deviation of f1 scoring vs C", "Values of C using Logistic", "F1 scoring mean")
    leg_point0 = mpl.Rectangle((0,0), 1, 1, fc = 'grey')
    leg_point1 = mpl.Rectangle((0,0), 1,1 , fc = 'blue')
    mpl.legend([leg_point0,leg_point1], ["f1 score mean", "Standard Deviation"])
    mpl.show()

    plot_error(degrees, secondPolyMeanArray, secondPolyStdArray,"grey","blue", "10-fold cross validation, f1 scoring vs degrees of Polynomial Features", "PolynomialFeatures(n)", "F1 scoring mean")
    leg_point0 = mpl.Rectangle((0,0), 1, 1, fc = 'grey')
    leg_point1 = mpl.Rectangle((0,0), 1,1 , fc = 'blue')
    mpl.legend([leg_point0,leg_point1], ["Standard Deviation", "f1 scoring mean"])
    mpl.show()

#Logistic Regression Classifier using best polyfeature and C Value.
    bestPolyFit = getPoly(2, secondX, secondTargetVal)
    model = LogisticRegression(penalty='l2', solver='lbfgs', C = 0.01)
    model.fit(bestPolyFit, secondTargetVal)
    ypred = model.predict(bestPolyFit)
    print(ypred)
    ax = mpl.subplot()
    trainColor = classifier(secondTargetVal, 'red', 'cornflowerblue')
    predictColour = classifier(ypred, 'black', 'lime')
    for labels in ["Trained data y = 1", "Trained data y = -1"]:
        ax.scatter(X3, X4, c = trainColor, label = labels)
    for labels in ["Predicted data y = 1", "Predicted data y = -1"]:
        ax.scatter(X3, X4, c = predictColour, marker = '+', label = labels, s= 5**2)
    mpl.xlabel("feature X1")
    mpl.ylabel("feature X2")
    mpl.title("Logistic Regression Classifier with polyfeature of degree 2 and C = 0.01")
    mpl.legend(loc = 'lower center')
    ax = mpl.gca()
    leg = ax.get_legend()
    leg.legendHandles[1].set_color("cornflowerblue")
    leg.legendHandles[2].set_color('black')
    leg.legendHandles[3].set_color('lime')
    mpl.show()

#nearest neighbors
    secondXTrain, secondXTest, secondTargetValTrain, secondTargetValTest = train_test_split(secondX, secondTargetVal, test_size=0.2)
    kVals = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    meanArrayK, stdArrayK = getBestK(kVals, secondXTrain, secondTargetValTrain)
    print(stdArrayK)
    meanPolyK, stdPolyK = getBestKAugmentPoly(kVals, secondXTrain, secondTargetValTrain, 2)
    meanPolyK1, stdPolyK1 = getBestKAugmentPoly(kVals, secondXTrain, secondTargetValTrain, 3)

    plot_error(kVals, meanArrayK, stdArrayK, "lime", 'red', "10-fold cross validation of kNN to determine best k", "k", "f1 scoring mean" )
    leg_point0 = mpl.Rectangle((0,0), 1, 1, fc = 'lime')
    leg_point1 = mpl.Rectangle((0,0), 1,1 , fc = 'red')
    mpl.legend([leg_point0,leg_point1], ["Standard Deviation", "f1 scoring mean"])
    mpl.show()
    plot_error(kVals, meanArrayK, stdArrayK, "lime", 'red', None, None, None )
    plot_error(kVals, meanPolyK1, stdPolyK1, "pink", 'green', None, None, None )
    plot_error(kVals, meanPolyK, stdPolyK, "grey", 'blue', None, None, None )
    mpl.title("kNN using no augmented and augmented polynomial features")
    mpl.xlabel("k = number of neighbors")
    mpl.ylabel("f1 scoring mean")
    mpl.legend(["no Polyfeatures", "PolyFeatures(2)", "PolyFeatures(3)"], loc='lower right')
    mpl.show()
    
    #using best value of k
    model = KNeighborsClassifier(n_neighbors=8, weights = 'uniform').fit(secondXTrain, secondTargetValTrain)
    secondYPred = model.predict(secondXTest)
    secondTargetValTest = np.asarray(secondTargetValTest)
    secondTargetValTrain = np.asarray(secondTargetValTrain)
    testColor = classifier(secondTargetValTest, 'red', 'cornflowerblue')
    predictColour = classifier(secondYPred, 'black', 'lime')
    for labels in ["Test data y = 1", "Test data y = -1"]:
        mpl.scatter(secondXTest[:,0], secondXTest[:,1], c = testColor, label = labels)
    for labels in ["Predicted Test data y = 1", "Predicted Test data y = -1"]:
        mpl.scatter(secondXTest[:,0], secondXTest[:,1], c = predictColour, marker = '+', label = labels)
    mpl.xlabel("feature X1")
    mpl.ylabel("feature X2")
    mpl.title("kNN Classifier k = 10 using Testing Data")
    mpl.gca()
    leg = mpl.legend()
    leg.legendHandles[0].set_color('red')
    leg.legendHandles[1].set_color('cornflowerblue')
    leg.legendHandles[2].set_color('black')
    leg.legendHandles[3].set_color('lime')
    mpl.show()

    #CONFUSTION MATRICES second dataset
    dummyModel = DummyClassifier(strategy="most_frequent").fit(secondXTrain, secondTargetValTrain)
    dummyPred = dummyModel.predict(secondXTest)

    print("Confusion Matrix for Logistic Regression Classifier")
    print(confusion_matrix(secondTargetVal, ypred))

    print("Confusion Matrix for kNN")
    print(confusion_matrix(secondTargetValTest, secondYPred))

    print("Confusion Matrix for Dummy Classifier")
    print(confusion_matrix(secondTargetValTest, dummyPred))

    #ROC Curves
#Logistic
    Xtrain, Xtest, ytrain, ytest = train_test_split(bestPolyFit, secondTargetVal, test_size=0.2)
    logisticModel = LogisticRegression(penalty='l2', solver='lbfgs', C = 1)
    logisticModel.fit(Xtrain, ytrain)
    fprLog, tprLog,_ = roc_curve(ytest, logisticModel.decision_function(Xtest))
    mpl.plot(fprLog, tprLog)
#kNN
    y_scores = model.predict_proba(secondXTest)
    fprknn, tprknn, _ = roc_curve(secondTargetValTest, y_scores[:,1])
    mpl.plot(fprknn, tprknn)
#baseline
    mpl.plot([0, 1], [0, 1], color='green',linestyle='--')
    mpl.xlabel("False Positive Rate")
    mpl.ylabel("True Positive Rate")
    mpl.title("ROC Curve with Logistic Regression and knn and baseline classifier")
    mpl.legend(["Logistic", "kNN", "Baseline"])
    mpl.show()


if __name__ == "__main__":
    main()