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


