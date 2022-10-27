import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
from sklearn.linear_model import LinearRegression


#function to normalise data
def normalise(list):
    mean = np.mean(list)
    std = np.std(list)
    for i in range(len(list)):
        list[i] = (list[i] - mean)/std


#set up values for gradient descent algorithm (change boolean is used to decide to change theta values for each iteration)
def gradient_descent(x, y, iterations, learning_rate, theta0, theta1, change):
    m = len(x)
    costT0T1 = [0] * iterations
    for i in range(iterations):
        #creating accumulators to add up the m number of elements in the list for each iteration.
        sumAcc = 0.0
        sumAcc0 = 0.0
        sumAcc1 = 0.0
        for j in range(m):
            hx = theta1 *x[j] + theta0 #predicted y value.
            cost = (1/m)* (hx - y[j])**2
            sumAcc = sumAcc+cost
            alpha0= (hx - y[j])
            sumAcc0 = sumAcc0+alpha0
            alpha1 = (hx - y[j])*x[j]
            sumAcc1 = sumAcc1 + alpha1
        #update theta values for next iteration
        alpha0 = -((2*learning_rate/m))*sumAcc0
        alpha1 = -((2*learning_rate/m))*sumAcc1
        #change = false if we are creating a baseline model.
        if(change == True):
            theta0 = theta0 + alpha0
            theta1 = theta1 + alpha1
        #adding value of cost function for that iteration to list.
        costT0T1[i] = sumAcc
    return costT0T1, theta0, theta1

#gradient descent in which the value of theta0 does not change

def main():
    df = pd.read_csv('week1.txt', comment='#')
    X = np.array(df.iloc[:,0], dtype=float); X=X.reshape(-1,1)
    y = np.array(df.iloc[:,1], dtype=float); y=y.reshape(-1,1)
    normalise(X)
    normalise(y)
    iterations = 300
    #plotting Gradient Descent algorithm
    costT0T1, theta0, theta1 = gradient_descent(X, y, iterations, 0.001, 0,  0, True)
    mpl.plot([j for j in range(iterations)], costT0T1)

    costT0T1, theta0, theta1 = gradient_descent(X, y, iterations, 0.01, 0,  0, True)
    mpl.plot([j for j in range(iterations)], costT0T1)

    costT0T1, theta0, theta1 = gradient_descent(X, y, iterations, 0.1, 0,  0, True)
    mpl.plot([j for j in range(iterations)], costT0T1)
    mpl.legend(["LR = 0.001", "LR = 0.01", "LR = 0.1"])
    mpl.xlabel("Number of Iterations"); mpl.ylabel("J(theta0, theta1).")
    mpl.title("Gradient Descent: Value of cost function given Learning Rate (LR).")
    mpl.show()

    #plotting linear regression using gradient descent
    costT0T1, theta0, theta1 = gradient_descent(X, y, iterations, 0.01, 0, 0, True)
    #print(float(theta0), float(theta1))
    mpl.scatter(X, y, marker='+', color = 'red')
    mpl.title("Linear Regression Using Gradient Descent with LR = 0.01")
    mpl.xlabel("input x")
    mpl.ylabel("output y")
    mpl.plot(X, ((theta1*X)+theta0))
    #mpl.show()
    #LR = 0.001
    # costT0T1, theta0, theta1 = gradient_descent(X, y, iterations, 0.001, 0, 0, True)
    # mpl.scatter(X, y, marker='+', color = 'red')
    # mpl.title("Linear Regression Using Gradient Descent with LR = 0.001")
    # mpl.xlabel("input x")
    # mpl.ylabel("output y")
    # mpl.plot(X, ((theta1*X)+theta0))
    # mpl.show()
    # #LR = 0.1
    # costT0T1, theta0, theta1 = gradient_descent(X, y, iterations, 0.1, 0, 0, True)
    # mpl.scatter(X, y, marker='+', color = 'red')
    # mpl.title("Linear Regression Using Gradient Descent with LR = 0.1")
    # mpl.xlabel("input x")
    # mpl.ylabel("output y")
    # mpl.plot(X, ((theta1*X)+theta0))
    # mpl.show()
    #LR = 0.01
    costT0T1, theta0, theta1 = gradient_descent(X, y, iterations, 0.01, 0, 0, False)
    #print(costT0T1)
    mpl.plot(X, ((theta1*X)+theta0))
    mpl.legend(["trained model","baseline model",  'training data'])
    mpl.title("Linear Regression Model using gradient descent with LR = 0.01")
    mpl.xlabel("input x")
    mpl.ylabel("output y")
    mpl.show()

    #plotting Linear Regression model using sklearn.
    regressionModel = LinearRegression().fit(X, y)
    print(regressionModel.coef_, regressionModel.intercept_)
    mpl.scatter(X, y, color = 'red', marker='+')
    mpl.plot(X, (regressionModel.coef_*(X) + regressionModel.intercept_), color = 'black')
    mpl.legend(["trained model",  'training data'])
    mpl.title("Linear Regression Model using sklearn")
    mpl.xlabel("input x")
    mpl.ylabel("output y")
    mpl.show()

if __name__ == "__main__":
    main()