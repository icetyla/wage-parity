import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data
path = os.getcwd()
train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')

train['overspend'] = (train.weekly_spending > train.allocated_wage).astype(int)

# Create variables for the model
X = train.drop(['index', 'first_name', 'last_name', 'agency', 'overspend'], axis=1)
X_test = test.drop(['index', 'first_name', 'last_name', 'agency'], axis=1)
y = train.overspend

# Split the training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

# Fit the model with the training data
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
preds = logistic.predict(X_valid)

# Assess the model with mean absolute error
mae_logs = mean_absolute_error(y_valid, preds)
print('MAE for Logisitic Regression:')
print(mae_logs)

# Create the final model
model = LogisticRegression()
model.fit(X, y)
final_preds = model.predict(X_test)

# Display predictions
output = pd.DataFrame({'first_name': test.first_name, 'last_name': test.last_name, 'overspend': final_preds})
print(output.head())