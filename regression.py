import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.over_sampling import SMOTE

# Load the data
path = os.getcwd()
train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')

# Build a label
train['overspend'] = (train.weekly_spending > train.allocated_wage).astype(int)

# Check for data imbalances
class_one = len(train[train.overspend == 0])
class_two = len(train[train.overspend == 1])

print('0: %d, 1: %d'%(class_one, class_two))

# Split dataset into variables and balance the label
X = train.drop(['index', 'first_name', 'last_name', 'agency', 'overspend'], axis=1)
X_test = test.drop(['index', 'first_name', 'last_name', 'agency'], axis=1)
y = train.overspend
sm = SMOTE()
X_sm, y_sm = sm.fit_resample(X, y)

print(y_sm.value_counts())


X_train, X_valid, y_train, y_valid = train_test_split(X_sm, y_sm, test_size=0.2, train_size=0.8, random_state=0)


# Fit the model
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
preds = logistic.predict(X_valid)

# Assess the model with accuracy, recall, and precision
accuracy = metrics.accuracy_score(y_valid, preds)
recall = metrics.recall_score(y_valid, preds)
precision = metrics.precision_score(y_valid, preds)
print('Accuracy score:', accuracy)
print('Recall score:', recall)
print('Precision score:', precision)

# Validate and visualize the model using the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_valid, preds)
roc_auc = metrics.auc(fpr, tpr)
graph = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
graph.plot()
plt.show()

# Create the final model
model = LogisticRegression()
model.fit(X, y)
final_preds = model.predict(X_test)

# Display predictions
output = pd.DataFrame({'first_name': test.first_name, 'last_name': test.last_name, 'overspend': final_preds})
print(output.head())