######Import Libraries####
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##### Reading data from file######
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.drop(["Salary"],axis=1)
y = dataset["Salary"]

####Take care of missing data
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

####Encode Categorical data(from string to vektor)
#Creating Dummy Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#create vertors instead of LabelEncoder for X[0] France=1,0,0 Germany=0,1,0 Spain=0,0,1 
ct  = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder="passthrough")
X = np.array( ct.fit_transform(X))

#Replace text values with numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


####Split data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


######Scaling the features########
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

###LinearRegression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
pred = regressor.predict(X_test)

##Polynomial Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
# print(lin_reg)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=10)
#Transform features as poly mode as feature.
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)


##SVR Regression
#Standard Scaler [-3,+3]
from sklearn.preprocessing import StandardScaler
#Create 2 different StandardScaler. 
#Because each of them has different mean/max/min
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

##TREE Regression 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

##Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

#Metrics # Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


##Plot Datas
##Training dataset visualization
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

##Test dataset visualization
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()