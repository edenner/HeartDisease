# Heart Disease Classification in R
### Objective: 
Build an accurate logistic regression model to classify patients with heart disease based off patients' demographics, lifestyle factors, and certain lab tests. 
<br>
### Overview:
The markdown file `heartdisease.md` provides code, explanations, visualizations, analysis, and details about the data set. <br> 
The file provides a step by step walkthrough of the entire machine learning process:
 1. pre-processing and formatting the data frame
 2. randomly splitting the data into training and testing sets
 3. exploratory analysis of variable distributions
 4. training a logistic regression model
 5. interpretation of model estimates and statistics
 6. determination of optimal threshold for classification that maximizes sensitivity (true positive rate) and specificity (true negative rate) 
 7. perform five fold cross validation on the training set to better understand model's predictive capabilities 
 8. evaluate model's actual accuracy on the test set <br>
<br>
### Packages:
- `dplyr`: data tidying and feature manipulation 
- `ggplot`: data visualizations and graphs 
- `rsample`: random split of data into training and testing sets. Cross validation of final model's performance 
- `tidymodels`: fit a logistic regression model through a recipe and workflow 
- `pROC`: generate Receiver Characteristic Operator plot, calculate area under the curve, sensitivity and specificity rates 
