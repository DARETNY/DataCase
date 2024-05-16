import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
data = pd.read_csv('/Users/namnam/Downloads/dataset.csv')

# Select the columns in the DataFrame that are of type float64
float_columns = data.select_dtypes(include=[np.float64]).columns

# Remove rows with missing values from the DataFrame
cleaned_data = data.dropna()

# Calculate the number of missing values in each float column
missing_float_data = data[float_columns].isnull().sum()
# Plot a pie chart for each feature in the dataset

# Print the number of missing values in each float column
print(f'missing values removed:\n {missing_float_data}')

# Print the total number of missing values
print(f'total missing values: {missing_float_data.sum()}')

# Save the cleaned data to a new CSV file
cleaned_data.to_csv('/Users/namnam/Downloads/cleaned_dataset.csv', index=True, header=True, sep=',', mode='w')


