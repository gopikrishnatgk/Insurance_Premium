# Importing required libraries
import pandas as pd

# Reading dataset
df = pd.read_csv('insurance.csv')

# Displaying top 5 rows
df.head()

## Depedent variable is continuous. So, it is a regression problem

# Displaying last 5 rows
df.tail()

# Displaying shape of Dataset i.e. showing no.of rows and no.of columns
print('No.of rows: ', df.shape[0])
print('No.of columns: ', df.shape[1])

# Information about our dataset. Displays No. of rows, No.of columns, Column headers, No. of Null, Datatypes           
df.info()

# Checking no. of null values in each column
df.isnull().sum()

# Statistical values of our dataset
# Shows total count, mean, standard devaiation, min, max etc., values of numerical columns
df.describe()

# Showing statistics of numerical and categorical columns
df.describe(include = 'all')

# Count plot of categorical data
import seaborn as sns

# Count plot of 'sex' column
sns.countplot(x = 'sex', data = df)

# Count plot of 'smoker' column
sns.countplot(x = 'smoker', data = df)

# Count plot of 'region' column
sns.countplot(x = 'region', data = df)

# Scaater plot of numerical data
import plotly.express as px

# Scatter plot for 'age' and 'bmi' columns
fig = px.scatter(df, x = 'age', y = 'bmi')
fig.show()

# Scatter plot for 'age' and 'children' columns
fig = px.scatter(df, x = 'age', y = 'children')
fig.show()

# Finding unique values from 'sex' column
df['sex'].unique()

# Assigning numerical values to categorical values fro 'sex' column
df['sex'] = df['sex'].map({'female':0, 'male':1})

# Checking unique values in 'smoker' column
df['smoker'].unique()

# Converting categorical values into numberical values
df['smoker'] = df['smoker'].map({'yes':1, 'no':0})

# Checking unique values in 'region' values
df['region'].unique()

# Converting categorical data into numerical data
df['region'] = df['region'].map({'southwest':1, 'southeast':2, 'northwest': 3, 'northeast':4})

# Checking the first 5 rows of data again
df.head()

# Defining Independent and Dependent variables
X = df.drop(['charges'], axis=1) # It is matrix
y = df['charges'] # It is a vector

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split     # Importing required libraries 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)  # 80% data to train and 20% data to test

# Showing X_train matrix
X_train

# Showing y_train vector
y_train

# Showing X_test matrix
X_test

# Showing y_test vector
y_test

# Importing models for regression
from sklearn.linear_model import LinearRegression  # For Linear Regression
from sklearn.svm import SVR  # For Support Vector Machine Regression
from sklearn.ensemble import RandomForestRegressor  # For Random Forest Regression
from sklearn.ensemble import GradientBoostingRegressor  # For Gradient Boosting Regression

# Training the model
lr = LinearRegression()  # Instance for Linear Regression
lr.fit(X_train, y_train)

svm = SVR()  # Instance for SVM
svm.fit(X_train, y_train)

rf = RandomForestRegressor()  # Instance for Random Forest Regression
rf.fit(X_train, y_train)

gr = GradientBoostingRegressor()  # Instanceo for Gradient Boosting Regression
gr.fit(X_train, y_train)

# Data Prediction on Test Data
y_lr = lr.predict(X_test)
y_svm = svm.predict(X_test)
y_rf = rf.predict(X_test)
y_gr = gr.predict(X_test)

# Assinging Data predictions using different regressions into a Data frame
df_pred = pd.DataFrame({'Actual':y_test,
                        'Lin.Reg.':y_lr,
                        'SVM':y_svm,
                        'Ran.Forest':y_rf,
                        'Grad.Boost.':y_gr})

# Displaying Data Predictions DataFrame
df_pred

# Performance of models through visualization
# Importing required libraries
import matplotlib.pyplot as plt

# Showing 4 plots at a time
plt.subplot(221)
plt.plot(df_pred['Actual'].iloc[0:11], label = 'Actual')
plt.plot(df_pred['Lin.Reg.'].iloc[0:11], label = 'Linear Reg')
plt.legend()

plt.subplot(222)
plt.plot(df_pred['Actual'].iloc[0:11], label = 'Actual')
plt.plot(df_pred['SVM'].iloc[0:11], label = 'SVM')
plt.legend()

plt.subplot(223)
plt.plot(df_pred['Actual'].iloc[0:11], label = 'Actual')
plt.plot(df_pred['Ran.Forest'].iloc[0:11], label = 'Random Forest')
plt.legend()

plt.subplot(224)
plt.plot(df_pred['Actual'].iloc[0:11], label = 'Actual')
plt.plot(df_pred['Grad.Boost.'].iloc[0:11], label = 'Gradient Boosting')
plt.legend()

plt.tight_layout()

## Out of 4 models Gradient Boosting Regression gave closer results with Actual values

# Evaluating the models using R square. R Square values is used to measure the goodness of fit
from sklearn import metrics

score_lr = metrics.r2_score(y_test, y_lr)  # For Lin. Reg. model
score_svm = metrics.r2_score(y_test, y_svm)  # For SVM model
score_rf = metrics.r2_score(y_test, y_rf)  # For Random Forest model
score_gr = metrics.r2_score(y_test, y_gr)  # For Gradient Boosting model

# Printing score of each regressor
print(score_lr, score_svm, score_rf, score_gr)

## Gradient Boosting Regression performing good than remaining 3 models

# Considering new data to predict with some testing values
new_data = {'age': 28,
           'sex':1,
           'bmi':42.1,
           'children':1,
            'smoker':0,
           'region':3}

new_df = pd.DataFrame(new_data, index = [0])
new_df

# Prediction with new data
pred = gr.predict(new_df)
print(pred)
