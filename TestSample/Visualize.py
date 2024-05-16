import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset into a pandas DataFrame
cleaned_data = pd.read_csv('/Users/namnam/Downloads/cleaned_dataset.csv')

# List of features in the dataset
features = ['feature_1', 'feature_2', 'feature_3', 'feature_4']

# For each feature in the dataset
for feature in features:
    # Plot a histogram of the feature
    plt.hist(cleaned_data[feature], bins=30, alpha=0.5)
    # Set the title of the plot as 'Histogram of {feature}'
    plt.title(f'Histogram of {feature}')
    # Draw a vertical line at the mean value of the feature
    plt.axvline(cleaned_data[feature].mean(), color='k', linestyle='dashed', linewidth=1)
    # Display the plot
    plt.show()

# Count the number of 'True' and 'False' in the 'isVirus' column
isVirus_counts = cleaned_data['isVirus'].value_counts()
# Plot a pie chart of the 'isVirus' counts
plt.pie(isVirus_counts, labels=['False', 'True'], autopct='%1.1f%%')
# Set the title of the plot as 'Pie chart of isVirus'
plt.title('Pie chart of isVirus')
# Display the plot
plt.show()

# Calculate the correlation matrix of the cleaned data
corr = cleaned_data.corr()

# Set the size of the figure to be 10x10
plt.figure(figsize=(10, 10))
# Create a heatmap of the correlation matrix
# The heatmap is annotated with the correlation coefficients formatted to 2 decimal places
# The colormap used is 'coolwarm'
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
# Set the title of the plot as 'Correlation Matrix'
plt.title('Correlation Matrix')
# Display the plot
plt.show()

defaultdata = pd.read_csv('/Users/namnam/Downloads/dataset.csv')

# user bar grap compare the two dataset
plt.bar(['Original', 'Cleaned'], [defaultdata.shape[0], cleaned_data.shape[0]])
plt.xlabel('Dataset')
plt.title('Number of Rows in Original and Cleaned Dataset')
plt.ylabel('Number of Rows')
plt.show()



