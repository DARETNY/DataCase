# Pear Inc. Smart T-shirt Antivirus Project

Welcome to the repository for Pear Inc.'s smart t-shirt antivirus proof of concept. Our goal is to create a lightweight model that can classify Google Play Store app signatures as either "Virus" or "Not a Virus". This model will eventually be implemented in our innovative smart t-shirts, enabling them to scan and approve apps on the fly.

## Project Overview

Pear Inc. is developing a groundbreaking smart t-shirt equipped with Bluetooth and customizable through downloadable applications. To ensure security and protect users from malicious apps, we need an antivirus model that can classify app signatures.

Our engineers have developed an algorithm called 'MAGIC' (manifold averaging generally intelligent compressor), which generates 4-dimensional numerical signatures for Google Play Store apps. Our task is to develop a model that takes these signatures as input and outputs labels (Virus or Not a Virus).
 

## Cleaned Dataset
This Python script uses pandas and numpy to clean a dataset. It first loads the dataset into a pandas DataFrame, then it selects the columns that are of type float64. It removes rows with missing values from the DataFrame and calculates the number of missing values in each float column. It then prints the number of missing values in each float column and the total number of missing values. Finally, it saves the cleaned data to a new CSV file.

We converted to some empty data set to cleaned_dataset
```
from:
dataset 2.csv
to:
cleaned_dataset.csv
```
New data set includes :
```
ID: 0, feature1, feature2, ..., isVirus
0, value1, value2, ..., 0
1, value1, value2, ..., 1
...

```



## Visualize 

This Python script uses pandas, matplotlib, and seaborn to visualize a cleaned dataset. It first loads the dataset into a pandas DataFrame, then it plots a histogram for each feature in the dataset, showing the distribution of the data and the mean value. It also plots a pie chart showing the proportion of 'True' and 'False' in the 'isVirus' column. Finally, it calculates the correlation matrix of the data and visualizes it as a heatmap, which can be used to understand the relationships between different features in the dataset.

 
## Screen Shoots

<a href="https://ibb.co/s6jNFc9"><img src="https://i.ibb.co/s6jNFc9/myplot.png" alt="myplot" border="0"></a>  
<a href="https://ibb.co/H7wjG0s"><img src="https://i.ibb.co/H7wjG0s/Histogram2.png" alt="Histogram2" border="0"></a> <a href="https://ibb.co/rkczXRM"><img src="https://i.ibb.co/rkczXRM/Histogram3.png" alt="Histogram3" border="0"></a> <a href="https://ibb.co/wr68ZTb"><img src="https://i.ibb.co/wr68ZTb/Histogram4.png" alt="Histogram4" border="0"></a> <a href="https://ibb.co/6Ng92sB"><img src="https://i.ibb.co/6Ng92sB/Piechart-Virus.png" alt="Piechart-Virus" border="0"></a> <a href="https://ibb.co/j8hNF6M"><img src="https://i.ibb.co/j8hNF6M/Summarize.png" alt="Summarize" border="0"></a>



## Test

By applying more advanced models and techniques, we were able to improve the performance of our virus detection model. The use of Random Forest and hyperparameter tuning significantly increased the accuracy and the AUC of the ROC curve.

<a href="https://ibb.co/LxWKMXW"><img src="https://i.ibb.co/LxWKMXW/Rog.png" alt="Rog" border="0"></a>

```bash
Before the Test:
  Accuracy=0.6338461538461538
After the Test:
  Accuracy: 0.8615384615384616

```
By applying more advanced models and techniques, we were able to improve the performance of our virus detection model. The use of Random Forest and hyperparameter tuning significantly increased the accuracy and the AUC of the ROC curve.




  <a href="https://ibb.co/yWmM6ct"><img src="https://i.ibb.co/yWmM6ct/Rog2.png" alt="Rog2" border="0"></a>

### Results 
* Accuracy: The accuracy of the initial Logistic Regression model and the improved Random Forest model.
* Confusion Matrix: A matrix showing the true positives, true negatives, false positives, and false negatives for both models.
* ROC Curve: A graph showing the performance of both models across different thresholds, with their respective AUC values.


## How to Run

* Clone the repository.
* Ensure you have the necessary libraries installed (e.g.,    `pandas`, `numpy, scikit-learn`, `matplotlib`).
* Run the provided scripts to visualize, clean, and train the model on the dataset.
* Evaluate the model using the evaluation scripts.

  
## To sum up

In this project, we developed a proof of concept for a lightweight antivirus model tailored for Pear Inc.'s innovative smart t-shirts. Using the 'MAGIC' algorithm, we processed app signatures and created a model to classify these signatures as either "Virus" or "Not a Virus".

Key Steps:
* Data Visualization: Initial exploration and visualization of the dataset to understand its structure.
* Data Cleaning: Handling missing values and balancing the dataset to improve model accuracy.
* Model Development: Creating a logistic regression model to classify app signatures.
* Model Evaluation: Assessing the model's performance using metrics like accuracy, ROC curve, and confusion matrix.


The result is a preliminary model that shows the potential for implementation in our smart t-shirts, ensuring that only safe, Pear Inc.-approved applications can be installed by users.



  

