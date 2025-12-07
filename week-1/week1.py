import numpy as np
import pandas as pd
#import needed libraries
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#load the data
df=pd.read_csv("C:/Users/Alsheefa/Desktop/romi/weintern/housing.csv")

#inspect the data
print(df.head)
print(df.info)
print(df.describe)
print(df.columns)

#clean the data
print(df.isnull().sum())

df=df.dropna()

df=df.drop_duplicates()

df.columns=df.columns.str.strip()

#encode the data
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
df['BT']=le.fit_transform(df['BT'])

#declaring X and Y variables
# X = all columns except the target
x = df.drop('MEDV', axis=1)

# y = the target column
y = df['MEDV']

#split data
x_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(X_test)

#training
model=LinearRegression()
model.fit(x_train_scaled,y_train)

#predictions
y_pred=model.predict(x_test_scaled)
print("predicted values: ",y_pred)

#calculate RMSE
rmse =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error (RMSE): ",rmse)

#visualizing
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

#plot

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect fit line
plt.show()




