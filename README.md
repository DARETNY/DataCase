# DataCase
 

## Cleaned Dataset

This Python script uses pandas and numpy to clean a dataset. It first loads the dataset into a pandas DataFrame, then it selects the columns that are of type float64. It removes rows with missing values from the DataFrame and calculates the number of missing values in each float column. It then prints the number of missing values in each float column and the total number of missing values. Finally, it saves the cleaned data to a new CSV file.



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
