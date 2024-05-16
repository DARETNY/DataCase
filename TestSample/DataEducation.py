import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
# The dataset is assumed to be a CSV file located at the specified path
data = pd.read_csv('/Users/namnam/Downloads/cleaned_dataset.csv')

# Split the data into features and target
# The features are all columns except the last one
# The target is the last column
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and test sets
# The test set size is 40% of the total data
# The random state is set to 302 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=96)

# Create a Logistic Regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy score of the model
# The accuracy is the proportion of correct predictions
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Plot the accuracy score
# The plot is a bar chart with the accuracy score
# The y-axis range is set from 0 to 1
# Plot the evaluation metrics
plt.figure(figsize=(10, 5))
bars = plt.bar(['Accuracy', 'Recall', 'F1 Score'], [accuracy, recall, f1])
plt.ylim([0, 1])
plt.ylabel('Score')

# Add labels to the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom')

plt.show()


#If the model doesn't perform well, there could be several reasons:
#The logistic regression model might be too simple for the dataset. You could try more complex models like decision trees, random forest, or neural networks.
#The features might need more preprocessing. You could try normalizing or standardizing the features, handling missing values, or encoding categorical variables.
#The dataset might be imbalanced. You could try oversampling the minority class, undersampling the majority class, or using a combination of both.
#The model might be overfitting or underfitting. You could try adjusting the model's hyperparameters, using cross-validation, or gathering more data.
