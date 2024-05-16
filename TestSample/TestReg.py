from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from Evolve import X_train, X_test, y_train, y_test  # Importing training and testing data from Evolve module

# Standardizing the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit to data, then transform it
X_test_scaled = scaler.transform(X_test)  # Perform standardization by centering and scaling

# Creating a Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Parameters for hyperparameter search
param_grid = {
    'n_estimators': [100, 200, 300],  # The number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # The maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # The minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # The minimum number of samples required to be at a leaf node
}

# Using GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)  # Fitting the model to the training data

# Using the best model to make predictions
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)  # Predicting the test results

# Calculating the accuracy of the new model
print('Accuracy:', metrics.accuracy_score(y_test, y_pred_rf))

# Calculating and plotting the ROC curve and AUC
fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test, y_pred_rf)
roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()