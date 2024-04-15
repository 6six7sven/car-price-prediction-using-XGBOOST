# %% [markdown]
# Importing libraries we will use for our car price prediction model.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# %%
data = pd.read_csv("F:\Code-Space\carPrice\Car-Price-Prediction\Car_extended.csv")

# %%
data.head()

# %%
data.count()

# %%
data.isnull().sum()

# %%
data[data["mileage"].isnull() | data["engine"].isnull() | data["seats"].isnull()]

# %%
data.dropna(inplace=True)

# %%
data['max_power'] = data['max_power'].apply(lambda x : x.split()[0] if type(x)==str else np.nan )
data['engine'] = data['engine'].apply(lambda x: x.replace("CC","") if type(x)==str else np.nan)
data['mileage'] = data['mileage'].apply(lambda x: float(x.split()[0]) if type(x)==str else np.nan)

# %%
data.drop(["torque"],axis=1,inplace=True)
data.drop(["name"],axis=1,inplace=True)

# %%
data["age"] = 2021-data["year"]
data.drop(["year"],axis=1,inplace=True)

# %%
data.shape

# %%
data.head()

# %%
data.describe()

# %%
Owners = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4}
data['owner'] = data['owner'].map(Owners)

# %%
data.tail()

# %%
sns.barplot(x='owner',y='selling_price',data=data)

# %%
sns.barplot(x='transmission',y='selling_price',data=data)

# %%
sns.barplot(x='seats',y='selling_price',data=data)

# %%
sns.barplot(x='fuel',y='selling_price',data=data)

# %%
sns.barplot(x='seller_type',y='selling_price',data=data)

# %%
sns.barplot(x='age',y='selling_price',data=data)

# %%
plt.scatter(data['km_driven'], data['selling_price'])
plt.xlabel("kms driven")
plt.ylabel("selling price")   
plt.show()

# %%
plt.scatter(data['mileage'], data['selling_price'])
plt.xlabel("mileage")
plt.ylabel("selling price")   
plt.show()

# %%
plt.scatter(data['engine'], data['selling_price'])
plt.xlabel("engine")
plt.ylabel("selling price")   
plt.show()

# %%
plt.scatter(data['max_power'], data['selling_price'])
plt.xlabel("max power")
plt.ylabel("selling price")   
plt.show()

# %%
data.head()

# %%
categorical_cols = ['fuel', 'seller_type', 'transmission'] 

data = pd.get_dummies(data, columns = categorical_cols)

# %%
data.head()

# %%
data.isnull().any()

# %%
data.dropna(subset = ["owner"], inplace=True)

# %%
data.isnull().any()

# %%
data.shape

# %%
correlations = data.corr()

indx=correlations.index
plt.figure(figsize=(20,12))
sns.heatmap(data[indx].corr(),annot=True,cmap="YlGnBu")

# %%
Y=data[['selling_price']]
Y.head()

# %%
X=data.drop(['selling_price'], axis=1)
X.head()

# %%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25, random_state=42)

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(Y_train)
Y_train = scaler.transform(Y_train)
Y_test = scaler.transform(Y_test)

# %%
from sklearn.linear_model import LinearRegression

linearRegression = LinearRegression().fit(X_train, Y_train)
y_pred=linearRegression.predict(X_test)

# %%
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
def rmse(ytrue, ypredicted):
    return np.sqrt(mse(ytrue, ypredicted))

# %%
rmse(Y_test, y_pred)

# %%
mae(Y_test,y_pred)

# %%
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
r2_score = regressor.score(X_test,Y_test)
print(r2_score*100,'%')

# %%
from xgboost import XGBRegressor
xgbr_2 = XGBRegressor(verbosity=0)
xgbr_2.fit(X_train, Y_train)
y_pred_X_2 = xgbr_2.predict(X_test)

# %%
mae(Y_test,y_pred_X_2)

# %%
rmse(Y_test, y_pred_X_2)

# %%
from sklearn.metrics import r2_score
print(r2_score(Y_test, y_pred_X_2)*100)


