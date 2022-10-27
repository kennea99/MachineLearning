import numpy as np
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

def readFiles():
    df = pd.read_csv("week3.txt", comment = '#')
    X1 = df.iloc[:,0]
    X2 = df.iloc[:,1]
    targetVal = df.iloc[:,2]
    X = np.column_stack((X1,X2))
    return X1, X2, targetVal, X

def alphaCalc(c):
    return 1/(2*c)

def lassoReg(c):
    clf = linear_model.Lasso(alpha=alphaCalc(c))
    return clf

def ridgeReg(c):
    clf = Ridge(alpha = alphaCalc(c))
    return clf

def gridCreate(X):
    grid = np.linspace(-2,2)
    for i in grid:
        for j in grid:
            X.append([i,j])
    X = np.array(X)
    return X

def kCrossValid(X, targetVal, C, k, regModel):
    kf = KFold(n_splits=k)
    arr = np.array([0]*k, dtype=float) 
    i = 0
    for train, test in kf.split(X):
        if regModel == "Ridge":
            cross_model = ridgeReg(C)
        elif regModel == "Lasso":
            cross_model = lassoReg(C)
        cross_model.fit(X[train], targetVal[train])
        ypred = cross_model.predict(X[test])
        #print("K-fold cross validation for C = 1.")
        #print("intercept: ",cross_model.intercept_, " coefficients: ", cross_model.coef_, "  mean squared-error: ", mean_squared_error(targetVal[test], ypred))
        arr[i] = mean_squared_error(targetVal[test], ypred)
        i = i + 1
    meanCross = arr.mean()
    varCross = arr.var()
    stdCross = arr.std()
    return meanCross, varCross, stdCross

def fold10_cVals(X, targetVal, cVals, regModel):
    meanArray = np.array([0]*len(cVals), dtype=float)
    stdArray = np.array([0]*len(cVals), dtype=float)
    i = 0
    for i in range(len(cVals)):
        if(regModel == "Lasso"):
            mean, var, std = kCrossValid(X, targetVal, cVals[i], 10, "Lasso")
        elif(regModel == "Ridge"):
            mean, var, std = kCrossValid(X, targetVal, cVals[i], 10, "Lasso")   
        meanArray[i] = mean
        stdArray[i] = std        

    return meanArray, stdArray

def plot_graph(xTest, X1, X2, targetVal,  pred, colors, title, xLabel, yLabel, zLabel):
    figure = mpl.figure()
    ax = figure.add_subplot(111, projection='3d')
    ax.plot_trisurf(xTest[:,0], xTest[:,1], pred, color= colors)
    ax.scatter(X1, X2, targetVal, marker = '.', color= 'r', label = "Training Data")
    ax.set_title(title)
    ax.set_label("Predicted Values")
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    #manually creating legend as 3D plots are not too easy to auto create them.
    surf_legend2 = mpl.Rectangle((0,0), 1, 1, fc = colors)
    surf_legend = mpl.Circle((0,0), 5,  fc = 'red')
    mpl.legend([surf_legend, surf_legend2], ["Training Data","Predicted Data"])

def plot_error(cVals, mean, std, colors, title, xLabel, yLabel):
    mpl.errorbar(cVals, mean, yerr=std, ecolor=colors)
    mpl.title(title)
    mpl.xlabel(xLabel)
    mpl.ylabel(yLabel)
    leg_point0 = mpl.Rectangle((0,0), 1, 1, fc = 'b')
    leg_point1 = mpl.Rectangle((0,0), 1,1 , fc = colors)
    mpl.legend([leg_point0,leg_point1], ["Mean", "Standard Deviation"])
    mpl.show()
    

def main():
    X1, X2, targetVal, X = readFiles()
    poly = PolynomialFeatures(5)
    fit = poly.fit_transform(X)
#(i)(a)
    fig = mpl.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1, X2, targetVal)
    ax.set_title("Downloaded Data Features and associated output")
    ax.set_xlabel("First Feature X1")
    ax.set_ylabel("Second Feature X2")
    ax.set_zlabel("Output y")
    mpl.show()


    #lasso regression with C = 1
    model0 = lassoReg(1)
    model0.fit(fit, targetVal)
    print("lasso regression with C = 1")
    print("coefficients: ", model0.coef_, " \n intercept: ", model0.intercept_)

    #lasso regression with C = 10
    model1 = lassoReg(10)
    model1.fit(fit, targetVal)
    print("lasso regression with C = 10")
    print("coefficients: ", model1.coef_, " \n intercept: ", model1.intercept_)

    #lasso regression with C = 1000
    model2 = lassoReg(1000)
    model2.fit(fit, targetVal)
    print("lasso regression with C = 1000")
    print("coefficients: ", model2.coef_, " \n intercept: ", model2.intercept_)

    #creating grid
    Xtest = []
    Xtest = gridCreate(Xtest)
    fitGrid = poly.fit_transform(Xtest)
    #predcited values with C = 1
    pred0 = model0.predict(fitGrid)
    plot_graph(Xtest, X1, X2, targetVal, pred0, 'grey', "Predictions Vs the actual Training Data with C = 1", "Feature X1", "Feature X2", "Output y")
    mpl.show()
    #predicted values with C = 10
    pred1 = model1.predict(fitGrid)
    plot_graph(Xtest, X1, X2, targetVal, pred1, 'grey', "Predictions Vs the actual Training Data with C = 10", "Feature X1", "Feature X2", "Output y")
    mpl.show()
    #predicted values with C = 1000
    pred2 = model2.predict(fitGrid)
    plot_graph(Xtest, X1, X2, targetVal, pred2, 'grey', "Predictions Vs the actual Training Data with C = 1000", "Feature X1", "Feature X2", "Output y")
    mpl.show()
    #print(fitGrid)

#RIDGE REGRESSION
    #Ridge regression with C = 0.01
    ridge_model0 = ridgeReg(0.01)
    ridge_model0.fit(fit, targetVal)
    print("Ridge regression with C = 0.01")
    print("coefficients: ", ridge_model0.coef_, " \n intercept: ", ridge_model0.intercept_)
    ridge_pred0 = ridge_model0.predict(fitGrid)
    plot_graph(Xtest, X1, X2, targetVal, ridge_pred0, 'grey', "Predictions vs training data with C = 0.01 using Ridge Regression", "Feature X1", "Feature X2", "Target Value y")
    mpl.show()

    #Ridge regression with C = 1
    ridge_model1 = ridgeReg(1)
    ridge_model1.fit(fit, targetVal)
    print("Ridge regression with C = 1")
    print("coefficients: ", ridge_model1.coef_, " \n intercept: ", ridge_model1.intercept_)
    ridge_pred1 = ridge_model1.predict(fitGrid)
    plot_graph(Xtest, X1, X2, targetVal, ridge_pred1, 'grey', "Predictions vs training data with C = 1 using Ridge Regression", "Feature X1", "Feature X2", "Target Value y")
    mpl.show()

    #Ridge regression with C = 1000
    ridge_model2 = ridgeReg(1000)
    ridge_model2.fit(fit, targetVal)
    print("Ridge regression with C = 1000")
    print("coefficients: ", ridge_model2.coef_, " \n intercept: ", ridge_model2.intercept_)
    ridge_pred2 = ridge_model2.predict(fitGrid)
    plot_graph(Xtest, X1, X2, targetVal, ridge_pred2, 'grey', "Predictions vs training data with C = 1000 using Ridge Regression", "Feature X1", "Feature X2", "Target Value y")
    mpl.show()
    
#CROSS VALIDATION QUESTIONS

    scores = cross_val_score(model0, X , targetVal, cv=5, scoring='neg_mean_squared_error')
    #print(scores)
    #kfold = 2
    meanCross2, varCross2, stdCross2 = kCrossValid(X, targetVal, 1, 2, "Lasso")
    #kfold = 5
    meanCross5, varCross5, stdCross5 = kCrossValid(X, targetVal, 1, 5, "Lasso")
    #kfold = 10
    meanCross10, varCross10, stdCross10 = kCrossValid(X, targetVal, 1, 10, "Lasso")
    #kfold = 25
    meanCross25, varCross25, stdCross25 = kCrossValid(X, targetVal, 1, 25, "Lasso")
    #kfold = 50
    meanCross50, varCross50, stdCross50 = kCrossValid(X, targetVal, 1, 50, "Lasso")
    #kfold = 100
    meanCross100, varCross100, stdCross100 = kCrossValid(X, targetVal, 1, 100, "Lasso")
    meanArray = np.array([meanCross2,meanCross5,meanCross10,meanCross25,meanCross50,meanCross100])
    varArray = np.array([varCross2,varCross5,varCross10,varCross25,varCross50,varCross100])
    folds = np.array([2,5,10,25,50,100])
    mpl.errorbar(folds, meanArray, yerr=varArray, ecolor='grey')
    mpl.title("kFold cross validation using the Lasso Model when C = 1")
    mpl.xlabel("number of folds")
    mpl.ylabel("Mean")
    leg_point0 = mpl.Rectangle((0,0), 1, 1, fc = 'b')
    leg_point1 = mpl.Rectangle((0,0), 1,1 , fc = 'grey')
    mpl.legend([leg_point0,leg_point1], ["Mean", "Variance"])
    mpl.show()

    cVals = np.array([1,10,20,40,60,80,100,200])
    meanArrayC, stdArrayC = fold10_cVals(X, targetVal, cVals, 'Lasso')
    plot_error(cVals, meanArrayC, stdArrayC,"grey", "10-fold cross validation, mean and  standard deviation vs C", "Values of C using Lasso", "Mean Values")

    meanArrayRidge, stdArrayRidge = fold10_cVals(X,targetVal,cVals, "Ridge")
    plot_error(cVals, meanArrayRidge, stdArrayRidge,"grey", "10-fold cross validation, mean and  standard deviation vs C using Ridge", "Values of C", "Mean Values")

if __name__ == "__main__":
    main()