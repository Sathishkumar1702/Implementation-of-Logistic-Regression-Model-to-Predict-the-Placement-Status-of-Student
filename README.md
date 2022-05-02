# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. start the program.
2. import pandas.
3. Read the dataset of placement status of student.
4. Drop a data of placement status
5. check isnull() and duplicate() of data given.
6. From sklearn import labelencoder, assign labelencoder as le.
7. assign a valiue x and y then print x and y.
8. from sklearn model import train_test_split.
9. from sklearn linear model import logistic regression.
10. from sklearn import accuracy_score and confuse_matrix.
11. stop the program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sathish kumar M 
RegisterNumber:  212220220042
*/
import pandas as pd
data=pd.read_csv("/content/sample_data/Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1.head()
x=data1.iloc[:,:-1]
print(x)
y=data1["status"]
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#library for larger linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](/O%201.png.png)
![the Logistic Regression Model to Predict the Placement Status of Student](/O%202.png.png)
![the Logistic Regression Model to Predict the Placement Status of Student](/O%203.png.png)
![the Logistic Regression Model to Predict the Placement Status of Student](/O%204.png.png)
![the Logistic Regression Model to Predict the Placement Status of Student](/O%205.png.png)
![the Logistic Regression Model to Predict the Placement Status of Student](/O%206.png.png)
![the Logistic Regression Model to Predict the Placement Status of Student](/O%207.png.png)
![the Logistic Regression Model to Predict the Placement Status of Student](/O%208.png.png)
![the Logistic Regression Model to Predict the Placement Status of Student](/O%209.png.png)
![the Logistic Regression Model to Predict the Placement Status of Student](/O%2010.png.png)
![the Logistic Regression Model to Predict the Placement Status of Student](/O%2011.png.png)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
