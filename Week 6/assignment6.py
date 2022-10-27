import numpy as np
import math
import pandas as pd
import random
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
from sklearn.kernel_ridge import KernelRidge



def genDummyData():
    dummyX = np.array([0] * 1000)
    for i in range(len(dummyX)):
        n = random.randint(-1,1)
        dummyX[i] = n
    dummyY = np.array([0] * len(dummyX))
    for j in range(len(dummyX)):
        if dummyX[j] == -1 or dummyX[j] == 1:
            dummyY[j] = 0
        elif dummyX[j] == 0:
            dummyY[j] = 1

    return dummyX, dummyY


def readFiles():
    df = pd.read_csv("week6.txt", comment = '#')
    X = np.array(df.iloc[:,0])
    Y = np.array(df.iloc[:,1])
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    return X,Y

def gridCreate(X):
    grid = np.linspace(-3,3)
    for i in grid:
        for j in grid:
            X.append([i,j])
    X = np.array(X)
    return X

def alphaCalc(c):
    return 1/(2*c)

def gaussian_kernel0(distances):
    weights = np.exp(-0*(distances**2))
    return weights

def gaussian_kernel1(distances):
    weights = np.exp(-1*(distances**2))
    return weights

def gaussian_kernel5(distances):
    weights = np.exp(-5*(distances**2))
    return weights

def gaussian_kernel10(distances):
    weights = np.exp(-10*(distances**2))
    return weights

def gaussian_kernel25(distances):
    weights = np.exp(-25*(distances**2))
    return weights

def gaussian_kernel50(distances):
    weights = np.exp(-50*(distances**2))
    return weights

def kernelRidgeReg(dummyX, dummyY, dummyXTest, C, g):
    model = KernelRidge(alpha = alphaCalc(C), kernel = 'rbf', gamma = g).fit(dummyX, dummyY)
    ypred = model.predict(dummyXTest)
    #print("parameter values for C = "+str(C) +" Gamma = "+str(g) +" : theta1: " +str(model.dual_coef_[0][0]) +", theta2: " +str(model.dual_coef_[1][0]) +", theta3: " +str(model.dual_coef_[2][0]))
    return model, ypred

def kCrossValidationKNN(X, Y, gamma, C):
    if gamma == 0:
        model = KNeighborsRegressor(n_neighbors=int(len(X)*0.8), weights = gaussian_kernel0)
    elif gamma == 1:
        model = KNeighborsRegressor(n_neighbors=int(len(X)*0.8), weights = gaussian_kernel1)
    elif gamma== 5:
        model = KNeighborsRegressor(n_neighbors=int(len(X)*0.8), weights = gaussian_kernel5)
    elif gamma == 10:
        model = KNeighborsRegressor(n_neighbors=int(len(X)*0.8), weights = gaussian_kernel10)
    elif gamma== 25:
        model = KNeighborsRegressor(n_neighbors=int(len(X)*0.8), weights = gaussian_kernel25)
    elif gamma == 50:
        model = KNeighborsRegressor(n_neighbors=int(len(X) *0.8), weights = gaussian_kernel50)
    scores = cross_val_score(model, X, Y, cv = 10, scoring="neg_mean_squared_error")
    #print(scores)
    return np.negative(scores.mean()), np.negative(scores.std())

def kCrossValidationRidgeGamma(X, Y, gammaVals):
    model = KernelRidge(alpha = alphaCalc(10), kernel='rbf', gamma=gammaVals)
    scores = cross_val_score(model, X, Y, cv = 10, scoring = 'neg_mean_squared_error')
    return np.negative(scores.mean()), np.negative(scores.std())

def kCrossValidationRidgeC(X, Y, cVals):
    model = KernelRidge(alpha = alphaCalc(cVals), kernel='rbf', gamma = 25)
    scores = cross_val_score(model, X, Y, cv = 10, scoring = 'neg_mean_squared_error')
    return np.negative(scores.mean()), np.negative(scores.std())

def kernelPlot(model, ypred, dummyX, dummyY, dummyXTest, gamma, colours):
    mpl.scatter(dummyX, dummyY, color = 'green')
    mpl.plot(dummyXTest, ypred, color = colours)


def knnPlot(dummyX, dummyY, dummyXTest, ypred, gamma, colours,titles):
    mpl.scatter(dummyX, dummyY, color = 'green')
    mpl.scatter(dummyXTest, ypred, marker = '+', c = colours, s =5**2)
    mpl.title(titles)
    mpl.xlabel("Input X")
    mpl.ylabel("Target Value Y")
    mpl.legend(["Training Data","Predicted Data: gamma = " +gamma])
    mpl.show()

def plot_error(cVals, mean, std, colors, lineColor, title, xLabel, yLabel):
    mpl.errorbar(cVals, mean, yerr=std, ecolor=colors, color = lineColor)
    mpl.title(title)
    mpl.xlabel(xLabel)
    mpl.ylabel(yLabel)


def main():
    X,Y = readFiles()

    #dummyX, dummyY = genDummyData()
    dummyX = np.array([-1, 0, 1])
    dummyY = np.array([0, 1, 0])

    colours = ['black', 'red', 'lawngreen', 'cyan', 'magenta']
    XTest = np.linspace(-3,3, num=len(X)).reshape(-1,1)
    dummyXTest = np.linspace(-3,3, num=100).reshape(-1,1)

    gamma = np.array([0, 1, 5, 10, 25])
    cVals = np.array([0.1, 1, 100])
    # print(dummyX)
    # print(dummyY)

    dummyX = dummyX.reshape(-1,1)
    dummyY = dummyY.reshape(-1,1)
    #dummyXTest = dummyXTest.reshape(-1,1)

# FOR (i)(a) and (i)(b)
    model0 = KNeighborsRegressor(n_neighbors=3, weights = gaussian_kernel0).fit(dummyX, dummyY)
    ypred0 = model0.predict(dummyXTest)

    model1 = KNeighborsRegressor(n_neighbors=3, weights = gaussian_kernel1).fit(dummyX, dummyY)
    ypred1 = model1.predict(dummyXTest)

    model5 = KNeighborsRegressor(n_neighbors=3, weights = gaussian_kernel5).fit(dummyX, dummyY)
    ypred5 = model5.predict(dummyXTest)

    model10 = KNeighborsRegressor(n_neighbors=3, weights = gaussian_kernel10).fit(dummyX, dummyY)
    ypred10 = model10.predict(dummyXTest)

    model25 = KNeighborsRegressor(n_neighbors=3, weights = gaussian_kernel25).fit(dummyX, dummyY)
    ypred25 = model25.predict(dummyXTest)

    knnPlot(dummyX, dummyY, dummyXTest, ypred0, '0', 'blue', "kNN Regression on Dummy Data")
    knnPlot(dummyX, dummyY, dummyXTest, ypred1, '1', 'red', "kNN Regression on Dummy Data")
    knnPlot(dummyX, dummyY, dummyXTest, ypred5, '5', 'brown', "kNN Regression on Dummy Data")
    knnPlot(dummyX, dummyY, dummyXTest, ypred10, '10', 'purple', "kNN Regression on Dummy Data")
    knnPlot(dummyX, dummyY, dummyXTest, ypred25, '25', 'black',"kNN Regression on Dummy Data")

#FOR (i)(c) and (i)(d)
    for i in range(len(cVals)):
        for j in range(len(gamma)):
            model, ypred = kernelRidgeReg(dummyX, dummyY, dummyXTest, cVals[i], gamma[j])
            kernelPlot(model, ypred, dummyX, dummyY, dummyXTest, gamma[j], colours[j])
        print("\n")
        mpl.title("Kernel Ridge Regression with C = " +str(cVals[i]))
        mpl.xlabel("input X")
        mpl.ylabel("output Y")
        mpl.legend(['Pred Value: g = '+str(gamma[j-4]), 'Pred Value: g = '+str(gamma[j-3]), 'Pred Value: g = '+str(gamma[j-2]), 'Pred Value: g = '+str(gamma[j-1]), 'Pred Value: g = '+str(gamma[j]), "Training Data"])
        mpl.show()
    #print(ypred)

#FOR (ii)(a)
    model0 = KNeighborsRegressor(n_neighbors=len(X), weights = gaussian_kernel0).fit(X, Y)
    ypred0 = model0.predict(XTest)

    model1 = KNeighborsRegressor(n_neighbors=len(X), weights = gaussian_kernel1).fit(X, Y)
    ypred1 = model1.predict(XTest)

    model5 = KNeighborsRegressor(n_neighbors=len(X), weights = gaussian_kernel5).fit(X, Y)
    ypred5 = model5.predict(XTest)

    model10 = KNeighborsRegressor(n_neighbors=len(X), weights = gaussian_kernel10).fit(X, Y)
    ypred10 = model10.predict(XTest)

    model25 = KNeighborsRegressor(n_neighbors=len(X), weights = gaussian_kernel25).fit(X, Y)
    ypred25 = model25.predict(XTest)

    knnPlot(X, Y, XTest, ypred0, '0', 'blue', "kNN Regression on Actual Data")
    knnPlot(X, Y, XTest, ypred1, '1', 'red', "kNN Regression on Actual Data")
    knnPlot(X, Y, XTest, ypred5, '5', 'brown', "kNN Regression on Actual Data")
    knnPlot(X, Y, XTest, ypred10, '10', 'purple', "kNN Regression on Actual Data")
    knnPlot(X, Y, XTest, ypred25, '25', 'black', "kNN Regression on Actual Data")

#FOR (ii)(b)

    for j in range(len(gamma)):
        model, ypred = kernelRidgeReg(X, Y, XTest, 1, gamma[j])
        kernelPlot(model, ypred, X, Y, XTest, gamma[j], colours[j])
    print("\n")
    mpl.title("Kernel Ridge Regression with C = 1")
    mpl.xlabel("input X")
    mpl.ylabel("output Y")
    mpl.legend(['Pred Value: g = '+str(gamma[j-4]), 'Pred Value: g = '+str(gamma[j-3]), 'Pred Value: g = '+str(gamma[j-2]), 'Pred Value: g = '+str(gamma[j-1]), 'Pred Value: g = '+str(gamma[j]), "Training Data"])
    mpl.show()
    
    gammaVals = np.array([0,1,5,10,25])
    cVals = np.array([0.1,1,10,25,50,100,250])
    knnMeanArray = np.array([0] * len(gammaVals), dtype=float)
    knnStdArray = np.array([0]* len(gammaVals), dtype=float)
#cross val for knn
    for i in range(len(gammaVals)):
        mean, std = kCrossValidationKNN(X, Y, gammaVals[i], cVals)
        print(std)
        knnMeanArray[i] = mean
        knnStdArray[i] = std
        #print(knnMeanArray)
    plot_error(gammaVals, knnMeanArray, knnStdArray, 'red', 'blue', '10-fold cross validation for gamma values on kNN', 'gamma', 'mean squared error')
    leg_point0 = mpl.Rectangle((0,0), 1, 1, fc = 'red')
    leg_point1 = mpl.Rectangle((0,0), 1,1 , fc = 'blue')
    mpl.legend([leg_point0,leg_point1], ["Standard Deviation", "mean squared error"])
    mpl.show()

#cross-val for gamma for ridge
    gammaRidgeMean = np.array([0] *len(gammaVals), dtype=float)
    gammaRidgeStd = np.array([0]*len(gammaVals), dtype=float)
    for i in range(len(gammaVals)):
        mean, std = kCrossValidationRidgeGamma(X, Y, gammaVals[i])
        gammaRidgeMean[i] = mean
        gammaRidgeStd[i] = std
    plot_error(gammaVals, gammaRidgeMean, gammaRidgeStd, 'red', 'blue', '10-fold cross validation for gamma values on Ridge with C = 10', 'gamma', 'mean squared error')
    leg_point0 = mpl.Rectangle((0,0), 1, 1, fc = 'red')
    leg_point1 = mpl.Rectangle((0,0), 1,1 , fc = 'blue')
    mpl.legend([leg_point0,leg_point1], ["Standard Deviation", "mean squared error"])
    mpl.show()

#cross val for C value for ridge gamma = 25
    cRidgeMean = np.array([0] * len(cVals), dtype=float)
    cRidgeStd = np.array([0] * len(cVals),dtype=float)
    for i in range(len(cVals)):
        mean, std = kCrossValidationRidgeC(X, Y, cVals[i])
        cRidgeMean[i] = mean
        cRidgeStd[i] = std
    plot_error(cVals, cRidgeMean, cRidgeStd, 'red', 'blue', '10-fold cross validation for C values on Ridge with gamma = 25', 'C Value', 'mean squared error')
    leg_point0 = mpl.Rectangle((0,0), 1, 1, fc = 'red')
    leg_point1 = mpl.Rectangle((0,0), 1,1 , fc = 'blue')
    mpl.legend([leg_point0,leg_point1], ["Standard Deviation", "mean squared error"])
    mpl.show()

#plotting the optimised 
    optKNNModel = KNeighborsRegressor(n_neighbors=len(X), weights=gaussian_kernel50).fit(X,Y)
    knnYPred = optKNNModel.predict(XTest)

    knnPlot(X,Y,XTest,knnYPred,'25', 'blue', 'Optimised KNN regression on Actual Data')


    optModel, optYPredRidge = kernelRidgeReg(X, Y, XTest, 10, 25)
    kernelPlot(optModel, optYPredRidge, X, Y, XTest, 25, 'red')
    mpl.title("Optimised Kernel Ridge Regression with C = 10 and Gamma = 25")
    mpl.xlabel("input x")
    mpl.ylabel("input y")
    mpl.legend(["Prediction", "Training Data"])
    mpl.show()

if __name__ == "__main__":
    main()