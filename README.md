# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.)Load and Preprocess Data: Read the dataset, drop unnecessary columns, and convert categorical variables into numerical codes using .astype('category') and .cat.codes.

2.)Define Variables: Split the dataset into features (X) and target variable (Y), and initialize a random parameter vector theta.

3.)Implement Functions: Define the sigmoid, loss, gradient_descent, and predict functions for logistic regression.

4.)Train Model: Use gradient descent to optimize the parameters theta over a specified number of iterations.

5.)Evaluate and Predict: Calculate accuracy of predictions on the training data, and demonstrate predictions with new sample data.

## Program:

### Program to implement the the Logistic Regression Using Gradient Descent.
#### Developed by: S DEEPIKA
#### RegisterNumber: 212223230039
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:/Users/admin/Downloads/Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

Y

theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 /(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred 

y_pred=predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

```

## Output:

### Dataset
![image](https://github.com/SanjayBalaji0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145533553/8040b2a6-d590-44c0-84a0-1ebeae4212c7)
### Dataset.dtypes
![image](https://github.com/SanjayBalaji0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145533553/1457c40d-7fa4-42cc-94c7-479fcdaf23cd)
### Dataset
![image](https://github.com/SanjayBalaji0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145533553/f5a9820d-9777-481c-8ba7-38d650ace909)
### Y
![image](https://github.com/SanjayBalaji0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145533553/29754610-b704-46ca-b12c-af7d82fa5b78)
### Accuracy
![image](https://github.com/SanjayBalaji0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145533553/e8d39e52-e857-4319-b59d-5bb652ef0967)
### Y_pred
![image](https://github.com/SanjayBalaji0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145533553/c58f3da5-0413-4fe2-9e14-45ba7d9426b0)
### Y
![image](https://github.com/SanjayBalaji0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145533553/e777bdf3-cccf-4376-b961-929223f78b09)
### Y_prednew
![image](https://github.com/SanjayBalaji0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145533553/ad1f7e8e-dec2-4190-a45a-1b53c9fe7a45)
### Y_prednew
![image](https://github.com/SanjayBalaji0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145533553/9416a246-2833-48a5-a123-1452d70a3da8)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

