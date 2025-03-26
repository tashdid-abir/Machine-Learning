# handling imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# model class
class myLR:
    
    def __init__(self):
        self.m = None
        self.b = None
        
    def fit(self, X_train, Y_train) -> tuple:
        
        num = 0
        den = 0
        
        for i in range(X_train.shape[0]):
            num = num + ((X_train[i] - X_train.mean())*(Y_train[i] - Y_train.mean()))
            den = den + ((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean()))
            
        self.m = num/den
        self.b = Y_train.mean() - (self.m * X_train.mean())
        
        return self.m, self.b
    
    def predict(self, X_test):
        return self.m * X_test + self.b
        

# creating the dataset
df = pd.DataFrame({'Hours Studied':[1,2,3,4,5], 'Exam Score':[50,55,65,70,75]})


#visualizing the data
# plt.scatter(df['Hours Studied'],df['Exam Score'])
# plt.title('Study vs Score')
# plt.xlabel('Hours Studied')
# plt.ylabel('Exam Score')
# plt.show()

# initializing the class
lr = myLR()

#fitting the model
m, b = lr.fit(df['Hours Studied'].values ,df['Exam Score'].values)
print(f"Slope is : {m} - Intercept is : {b}")

#predicting the score
score = lr.predict(7)
print(f"Score : {score}")

#calculating the r2 score
r2score = r2_score(df['Exam Score'], lr.predict(df['Hours Studied']))
print(f"R2 score : {r2score}")