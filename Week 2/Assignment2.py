import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def readFiles():
    df = pd.read_csv("week2.txt", comment = '#')
    X1 = df.iloc[:,0]
    X2 = df.iloc[:,1]
    X = np.column_stack((X1,X2))
    y = df.iloc[:,2]
    return X1, X2, X, y

#function to generate colour arrays to use to compare predicted target values to actual target values.
def classifier(yPredict, color1, color2):
    predictColours = [color1] * len(yPredict)
    for i in range(len(yPredict)):
        if(yPredict[i] == -1):
            predictColours[i] = color2
    return predictColours

def get_boundary(X1, X2, intercept, coefficient, decision_boundary):
    decision_boundary = -((intercept*1) + (coefficient[0][0] * X1))/coefficient[0][1]
    return decision_boundary

def squareFeatures(X1, X2):
    X3 = [0] * len(X1)
    X4 = [0] * len(X1)
    for i in range(len(X1)):
        X3[i] = (X1[i]**2)
    for j in range(len(X2)):
        X4[j] = (X2[j]**2)
    x_square = np.column_stack((X1,X2,X3,X4))
    return x_square, X3, X4

#attempting to construct a decision boundary for the square features but couldn't get it to work :(
def get_boundary_square(X1, X2, X3, X4, intercept, coefficient, square_boundary):
    square_boundary = -((intercept*1) + (coefficient[0][0] * X1) + (coefficient[0][2] * X3)+ (coefficient[0][3]*X4))/coefficient[0][1]
    return square_boundary

def firstPlot(X1, X2, pointColours):
    for labels in ["target value y = 1", "target value y = -1"]:
        mpl.scatter(X1, X2, c = pointColours, marker = '+', label = labels)
    mpl.title("Colour of each feature pair given y = 1 or y = -1")
    mpl.xlabel("X1")
    mpl.ylabel("X2")
    mpl.legend()
    mpl.gca()
    legend = mpl.legend()
    legend.legendHandles[1].set_color('cornflowerblue')
    mpl.show()


def plot(X1, X2, decision_boundary, predictColours, pointColours):
    ax = mpl.subplot()
    for labels in ["Trained data y = 1", "Trained data y = -1"]:
        ax.scatter(X1, X2, color=pointColours, label = labels)
    for labels1 in ["Predicted Data y = 1", "Predicted Data y = -1"]:
        ax.scatter(X1, X2, color=predictColours, marker = '+', label = labels1, s = 5**2)
    ax.plot(X1, decision_boundary, label = "decision_boundary", c = 'brown')
    mpl.legend()
    ax = mpl.gca()
    leg = ax.get_legend()
    leg.legendHandles[2].set_color('cornflowerblue')
    leg.legendHandles[4].set_color('black')
    mpl.ylabel("X2")
    mpl.xlabel("X1")
    mpl.title("Comparing Training Data with Predictions from Logistic Regression")
    mpl.show()

def svm_plot0(X1, X2, decision_boundary, predictColours, pointColours):
    ax = mpl.subplot()
    for labels in ["Trained data y = 1", "Trained data y = -1"]:
        ax.scatter(X1, X2, color=pointColours, label = labels)
    for labels1 in ["Predicted Data y = 1", "Predicted Data y = -1"]:
        ax.scatter(X1, X2, color=predictColours, marker = '+', label = labels1, s = 5**2)
    ax.plot(X1, decision_boundary, label = "decision_boundary", c = 'brown')
    mpl.legend()
    ax = mpl.gca()
    leg = ax.get_legend()
    leg.legendHandles[2].set_color('cornflowerblue')
    leg.legendHandles[4].set_color('black')
    mpl.ylabel("X2")
    mpl.xlabel("X1")
    mpl.title("Comparing Training Data with Predictions from SVMs with C = 0.001")
    mpl.show()

def svm_plot1(X1, X2, decision_boundary, predictColours, pointColours):
    ax = mpl.subplot()
    for labels in ["Trained data y = 1", "Trained data y = -1"]:
        ax.scatter(X1, X2, color=pointColours, label = labels)
    for labels1 in ["Predicted Data y = 1", "Predicted Data y = -1"]:
        ax.scatter(X1, X2, color=predictColours, marker = '+', label = labels1, s = 5**2)
    ax.plot(X1, decision_boundary, label = "decision_boundary", c = 'brown')
    mpl.legend()
    ax = mpl.gca()
    leg = ax.get_legend()
    leg.legendHandles[2].set_color('cornflowerblue')
    leg.legendHandles[4].set_color('black')
    mpl.ylabel("X2")
    mpl.xlabel("X1")
    mpl.title("Comparing Training Data with Predictions from SVMs with C = 0.01")
    mpl.show()

def svm_plot2(X1, X2, decision_boundary, predictColours, pointColours):
    ax = mpl.subplot()
    for labels in ["Trained data y = 1", "Trained data y = -1"]:
        ax.scatter(X1, X2, color=pointColours, label = labels)
    for labels1 in ["Predicted Data y = 1", "Predicted Data y = -1"]:
        ax.scatter(X1, X2, color=predictColours, marker = '+', label = labels1, s = 5**2)
    ax.plot(X1, decision_boundary, label = "decision_boundary", c = 'brown')
    mpl.legend()
    ax = mpl.gca()
    leg = ax.get_legend()
    leg.legendHandles[2].set_color('cornflowerblue')
    leg.legendHandles[4].set_color('black')
    mpl.ylabel("X2")
    mpl.xlabel("X1")
    mpl.title("Comparing Training Data with Predictions from SVMs with C = 1000")
    mpl.show()

def square_plot(X1, X2, square_pred_colours, pointColours):
    for labels in ["Actual data y = 1", 'Actual data y = -1']:
        mpl.scatter(X1, X2, c = pointColours, label = labels)
    for labels in ["Predicted data y = 1", 'Predicted data y = = -1']:
        mpl.scatter(X1, X2, c = square_pred_colours, label = labels, marker = "+", s  =5**2)
    mpl.xlabel("X1")
    mpl.ylabel("X2")
    mpl.title("Comparing predicted target values of squared features to actual target values")
    mpl.legend()
    mpl.gca()
    legend = mpl.legend()
    legend.legendHandles[1].set_color('cornflowerblue')
    legend.legendHandles[3].set_color('black')


    mpl.show()

def main():
    X1, X2, X, y = readFiles()
    pointColours = ['red'] * len(y)
    for i in range(len(pointColours)):
        if y[i] == -1:
            pointColours[i] = 'cornflowerblue'

    firstPlot(X1, X2, pointColours)
#FOR ASSIGNMENT (A) LOGISTIC REGRESSION
    model = LogisticRegression(penalty = 'none', solver='lbfgs')
    model.fit(X, y)
    yPredict = model.predict(X)
    predictColours = classifier(yPredict, 'lime', 'black')
    print("Parameter Values for a(ii):")
    print("Intercept (theta0): " +str(model.intercept_) + "Coefficients (theta1 and theta2): " +str(model.coef_) +'\n')
    decision_boundary = [0] * len(X2)
    decision_boundary = get_boundary(X1, X2, model.intercept_, model.coef_, decision_boundary)
    plot(X1, X2, decision_boundary, predictColours, pointColours)

#FOR PART (B) USING SVM'S
    svm_model0 = LinearSVC(C = 0.001).fit(X, y)
    print("SVM with C = 0.001")
    print("Intercept (theta0): " +str(svm_model0.intercept_) + "Coefficients (theta1 and theta2): " +str(svm_model0.coef_) +'\n')
    svm_decision0 = [0] * len(X2)
    svm_decision0 = get_boundary(X1, X2, svm_model0.intercept_, svm_model0.coef_, svm_decision0)
    svm_predict0 = svm_model0.predict(X)
    #print(svm_predict0)
    svm_pred_color0 = classifier(svm_predict0, 'lime', 'black')
    svm_plot0(X1, X2, svm_decision0, svm_pred_color0, pointColours)

    svm_model1 = LinearSVC(C = 0.01).fit(X, y)
    print("SVM with C = 0.01")
    print("Intercept (theta0): " +str(svm_model1.intercept_) + "Coefficients (theta1 and theta2): " +str(svm_model1.coef_) + "\n")
    svm_decision1 = [0] * len(X2)
    svm_decision1 = get_boundary(X1, X2, svm_model1.intercept_, svm_model1.coef_, svm_decision1)
    svm_predict1 = svm_model1.predict(X)
    # print(svm_predict1)
    svm_pred_color1 = classifier(svm_predict1, 'lime', 'black')
    svm_plot1(X1, X2, svm_decision1, svm_pred_color1, pointColours)

    test_svm_model2 = LinearSVC(C = 1).fit(X, y,)
    print("SVM with C = 1")
    print("Intercept (theta0): " +str(test_svm_model2.intercept_) + "Coefficients (theta1 and theta2): " +str(test_svm_model2.coef_) + "\n")

    svm_model2 = LinearSVC(C = 1000, max_iter=100000).fit(X, y,)
    print("SVM with C = 1000")
    print("Intercept (theta0): " +str(svm_model2.intercept_) + "Coefficients (theta1 and theta2): " +str(svm_model2.coef_) + "\n")
    svm_decision2 = [0] * len(X2)
    svm_decision2 = get_boundary(X1, X2, svm_model2.intercept_, svm_model2.coef_, svm_decision2)
    svm_predict2 = svm_model2.predict(X)
    #print(svm_predict2)
    svm_pred_color2 = classifier(svm_predict2, 'lime', 'black')
    svm_plot2(X1, X2, svm_decision2, svm_pred_color2, pointColours)

#FOR PART (C)
    #squaring each X1 and X2 value and adding it to an array.
    X3 = [0] * len(X1)
    X4 = [0] * len(X2)
    x_square, X3, X4 = squareFeatures(X1, X2)
    square_model = LogisticRegression(penalty = 'none', solver='lbfgs')
    square_model.fit(x_square, y)
    print("Parameter Values for c(i):")
    print("Intercept (theta0): " +str(square_model.intercept_) + "Coefficients (theta1, theta2, theta3, theta4): " +str(square_model.coef_) +'\n')
    #getting the predicted values
    square_predict = square_model.predict(x_square)
    square_pred_colours = classifier(square_predict, 'lime', 'black')
    #square_boundary = [0] * len(X2)
    square_plot(X1, X2, square_pred_colours, pointColours)
    # square_boundary = get_boundary_square(X1, X2, X3, X4, square_model.intercept_, square_model.coef_, square_boundary)
    # print(square_boundary)
    

if __name__ == "__main__":
    main()


