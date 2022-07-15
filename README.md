# Heart Disease Classification in R
### Objective: 
Build an accurate logistic regression model to classify patients with heart disease based off patients' demographics, lifestyle factors, and certain lab tests. 
<br>
### Overview:
The markdown file `heartdisease.md` provides code, explanations, visualizations, analysis, and details about the data set. 1. The file provides a step by step walkthrough of the entire machine learning process:
2. pre-processing and formatting the data frame
3. randomly splitting the data into training and testing sets
4. exploratory analysis of variable distributions
5. training a logistic regression model
6. interpretation of model estimates and statistics
7. determination of optimal threshold for classification that maximizes sensitivity (true positive rate) and specificity (true negative rate) 
8. perform five fold cross validation on the training set to better understand model's predictive capabilities 
9. evaluate model's actual accuracy on the test set 
<br>
### Packages:
- `dplyr`: data tidying and feature manipulation 
- `ggplot`: data visualizations and graphs 
- `rsample`: random split of data into training and testing sets. Cross validation of final model's performance 
- `tidymodels`: fit a logistic regression model through a recipe and workflow 
- `pROC`: generate Receiver Characteristic Operator plot, calculate area under the curve, sensitivity and specificity rates 
