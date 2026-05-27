# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import pandas module and import the required data set.
Find the null values and count them.
Count number of left values.
From sklearn import LabelEncoder to convert string values to numerical values.
From sklearn.model_selection import train_test_split.
Assign the train dataset and test dataset.
From sklearn.tree import DecisionTreeClassifier.
Use criteria as entropy.
From sklearn import metrics.
Find the accuracy of our model and predict the require values

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ETHIC VARAN S
RegisterNumber:  212224230072
*/

import pandas as pd
data=pd.read_csv("Employee.csv")

print("data.head():")
data.head()


print("data.info():")
data.info()


print("isnull() and sum():")
data.isnull().sum()


print("data value counts():")
data["left"].value_counts()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()


print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()


y=data["left"]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy value:", accuracy)


print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```

## Output:

<img width="897" height="747" alt="Screenshot 2026-05-26 135403" src="https://github.com/user-attachments/assets/44b3e691-d330-47d4-b197-50fabe2c1f46" />

<img width="1013" height="753" alt="Screenshot 2026-05-26 135418" src="https://github.com/user-attachments/assets/187e84c4-1863-42a7-971e-0c4f55097abd" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
