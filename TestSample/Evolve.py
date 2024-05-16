import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
# The dataset is assumed to be a CSV file with columns 'Unnamed: 0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', and 'isVirus'.
model = pd.read_csv('/Users/namnam/Downloads/cleaned_dataset.csv')

# Drop the 'Unnamed: 0' column as it seems to be an index
model = model.drop('Unnamed: 0', axis=1)

# Split the data into features and target
# The features are 'feature_1', 'feature_2', 'feature_3', 'feature_4'.
# The target is 'isVirus', which is a boolean indicating whether a certain instance is a virus or not.
X = model.drop('isVirus', axis=1)
y = model['isVirus']

# Split the dataset into training set and test set
# The test set is 20% of the total dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
# This model is used for binary classification problems.
log_reg = LogisticRegression()

# Train the model using the training set
log_reg.fit(X_train, y_train)

# Make predictions using the test set
y_pred = log_reg.predict(X_test)

# Print the accuracy of the model
# Accuracy is the proportion of true results among the total number of cases examined.
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

# Compute the confusion matrix
# A confusion matrix is a table that is often used to describe the performance of a classification model.
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Not Virus', 'Virus'], rotation=45)
plt.yticks(tick_marks, ['Not Virus', 'Virus'])

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Compute ROC curve and ROC area
# The ROC curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system.
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()