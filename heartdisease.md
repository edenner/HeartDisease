Heart Disease Prediction
================
Elena Denner

### Background and Objectives

Heart disease is the leading cause of death in the US, causing 25% of
deaths each year. The term heart disease can refer to a multitude of
heart conditions that put someone at a higher risk of heart attack. The
most common type of heart disease is coronary artery disease, in which
blood flow to the heart is restricted. Certain medical conditions and
lifestyle factors can put someone at a higher risk of heart disease,
such as diabetes, obesity, physical inactivity, diet, smoking, and
alcohol consumption (“Heart Disease”). The objectives of this project
are to build a machine learning model that accurately predicts if a
patient has heart disease or not based on their demographics and to
determine the most significant risk factors for heart disease.

### Load libraries

``` r
library(dplyr)
library(ggplot2)
library(forcats)
library(rsample)
library(tidyverse)
library(tidymodels)
library(gridExtra)
library(pROC)
```

### Read in data

Data was drawn from the UCI machine learning repository. The data
collection took place at 4 different medical clinics, and the Cleveland
clinic database is used here. There are 303 patients in the data set and
14 measurements provided for each patient: 5 measurements of heart
activity, 8 demographics/risk factors, and the response variable (heart
disease). The 8 demographic risk factors and their relationship to heart
disease are the focus of this project.

``` r
setwd('/Users/elena/Desktop/HeartDisease/HeartDisease/')
cleveland <- read.csv("cleveland.data", header = FALSE)
```

### Data Tidying

Before beginning analysis, the data must be cleaned. The column names
are manually entered and the naming scheme comes from the publishers of
the original data set. The 8 variables used for analysis are: sex (male
or female), age (years), chest pain type (typical angina, atypical
angina, non-angina pain, asymptomatic), fasting blood sugar (high if
measurement is \>120 mg/dl or low if measurement is \<120 mg/dl),
resting blood pressure (measured in mm Hg), serum cholesterol (measured
in mg/dl), maximum heart rate during exercise (beats per minute), and
exercise induced angina (yes or no). The variables sex, chest pain type,
fasting blood sugar, and exercise induced angina must be converted into
factors. In addition, the response variable for heart disease was
initially coded as an integer (0,1,2,3,4) and was re-coded into
‘absence’ (0) versus ‘presence’ (1,2,3,4). It was recommended by the
publishers of the data set to re-code the response variable for analysis
purposes. There were no missing values in the dataset and so all 303
observations were utilized for analysis.

``` r
names = c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", 
          "exang", "oldpeak", "slope", "ca", "thal", "heart_disease")
colnames(cleveland) <- names

cleveland <- cleveland %>%
  mutate(sex = case_when(sex == 0 ~ "female",
                         sex == 1 ~ "male")) %>%
  mutate(cp = case_when(cp == 1 ~ "typical angina",
                        cp == 2 ~ "atypical angina", 
                        cp == 3 ~ "non-anginal pain",
                        cp == 4 ~ "asymptomatic")) %>%
  mutate(fbs = case_when(fbs == 1 ~ "high",
                         fbs == 0 ~ "low")) %>% 
  mutate(exang = case_when(exang == 0 ~ "no",
                           exang == 1 ~ "yes")) %>%
  mutate(heart_disease = case_when(heart_disease == 0 ~ "absence",
                            TRUE ~ "presence"))

cleveland <- cleveland %>%
  mutate(sex = as.factor(sex)) %>%
  mutate(cp = as.factor(cp)) %>%
  mutate(fbs = as.factor(fbs)) %>%
  mutate(exang = as.factor(exang)) %>%
  mutate(heart_disease = as.factor(heart_disease))

cleveland <- cleveland %>%
  select(age, sex, cp, trestbps, chol, fbs, thalach, exang, heart_disease) %>%
  rename("max_hr" = "thalach",
         "exercise_angina" = "exang") %>%
  drop_na()

glimpse(cleveland)
```

    Rows: 303
    Columns: 9
    $ age             <dbl> 63, 67, 67, 37, 41, 56, 62, 57, 63, 53, 57, 56, 56, 44…
    $ sex             <fct> male, male, male, male, female, male, female, female, …
    $ cp              <fct> typical angina, asymptomatic, asymptomatic, non-angina…
    $ trestbps        <dbl> 145, 160, 120, 130, 130, 120, 140, 120, 130, 140, 140,…
    $ chol            <dbl> 233, 286, 229, 250, 204, 236, 268, 354, 254, 203, 192,…
    $ fbs             <fct> high, low, low, low, low, low, low, low, low, high, lo…
    $ max_hr          <dbl> 150, 108, 129, 187, 172, 178, 160, 163, 147, 155, 148,…
    $ exercise_angina <fct> no, yes, yes, no, no, no, no, yes, no, yes, no, no, ye…
    $ heart_disease   <fct> absence, presence, presence, absence, absence, absence…

### Exploratory Analysis

Next visualizations of each variable are created in order to familiarize
ourselves with the predictors. A faceted histogram shows the
relationship between heart disease and age while a bar chart is shown
for the relationship between heart disease and chest pain type.
Proportional bar charts are created for our 3 binary variables: sex,
fasting blood sugar, and exercise induced angina. Box plots are shown
for our 3 numerical variables: resting blood pressure, serum
cholesterol, and maximum heart rate.

``` r
age.plot <- ggplot(cleveland, mapping = aes(x = age, fill = heart_disease)) +
  geom_histogram() +
  facet_wrap(vars(heart_disease)) +
  labs(title = "Prevelance of Heart Disease Across Age", 
       x = "Age (years)", y = "Count", fill = "Heart Disease")
age.plot
```

![](heartdisease_files/figure-gfm/age-plot-1.png)<!-- -->

The histograms for age faceted on the presence and absence of heart
disease have different distribution shapes, suggesting that age does
have a relationship with heart disease. The distribution of the presence
of heart disease is left skewed while the distribution of the absence of
heart disease appears more normally distributed. These graphics suggests
that there are more older people with heart disease than younger people
with heart disease.

``` r
cp.plot <- ggplot(cleveland, mapping = aes(x=heart_disease, fill = cp)) +
  geom_bar(position = "dodge") +
  labs(title = "Prevelance of Heart Disease for Different Chest Pain Types", 
       x = "Heart Disease", y = "Count", fill = "Chest Pain Type")
cp.plot
```

![](heartdisease_files/figure-gfm/cp-plot-1.png)<!-- -->

There does appear to be a relationship between type of chest pain and
heart disease. Interestingly, asymptomatic chest pain type has the
highest count for the presence of heart disease, while typical angina
pain has the lowest count. There is a higher count of people without
heart disease that have atypical or typical angina chest pain compared
to people with heart disease. Angina is listed as one of the most common
symptoms of heart attack and so this result is skeptical and needs
further investigation, but we will assume it is correct for the current
analysis.

``` r
 sex.plot <- ggplot(cleveland, mapping = aes(x = sex, fill = heart_disease)) +
  geom_bar(position = "fill") +
  labs(x = "Sex", y = "Proportion", fill = "Heart Disease") +
  theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), 
        axis.title.y = element_text(size = 12), axis.text.y = element_text(size = 12))

fbs.plot <- ggplot(cleveland, mapping = aes(x=fbs, fill=heart_disease)) +
  geom_bar(position = "fill") +
  labs(x = "Fasting Blood Sugar", y = "Proportion", fill = "Heart Disease") +
  scale_x_discrete(labels = c("low", "high"))+
  theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), 
        axis.title.y = element_text(size = 12), axis.text.y = element_text(size = 12))

exang.plot <- ggplot(cleveland, mapping = aes(x = exercise_angina, fill = heart_disease)) +
  geom_bar(position = "fill") +
  labs(x = "Exercise induced angina", y = "Proportion", fill = "Heart Disease") +
  theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12))

bplots <- grid.arrange(sex.plot, fbs.plot, exang.plot, nrow=2)
```

![](heartdisease_files/figure-gfm/bp-plots-1.png)<!-- -->

``` r
bplots
```

    TableGrob (2 x 2) "arrange": 3 grobs
      z     cells    name           grob
    1 1 (1-1,1-1) arrange gtable[layout]
    2 2 (1-1,2-2) arrange gtable[layout]
    3 3 (2-2,1-1) arrange gtable[layout]

The bar plot on the top left shows a higher proportion of males with
heart disease than females with heart disease, suggesting that there is
a relationship between sex and heart disease. There is an even larger
distinction for heart disease as it relates to exercise induced angina,
for a much higher proportion of people with exercise induced angina have
heart disease compared to people without exercise induced angina (bottom
left plot). Fasting blood sugar levels do not appear to have a
correlation with heart disease, as there appears to be a similar
proportion of presence and absence of heart disease for people with high
and low fasting blood sugar levels (top right plot).

``` r
trestbps.plot <- ggplot(cleveland, mapping = aes(x=trestbps, y=heart_disease)) +
  geom_boxplot() +
  labs(x = "Resting Blood Pressure (mm Hg)", y = "Heart Disease") +
  theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), 
        axis.title.y = element_text(size = 12), axis.text.y = element_text(size = 12))

chol.plot <- ggplot(cleveland, mapping = aes(x=chol, y=heart_disease)) +
  geom_boxplot() +
  labs(x = "Serum Cholestoral (mg/dl)", y = "Heart Disease") +
  theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), 
        axis.title.y = element_text(size = 12), axis.text.y = element_text(size = 12))

maxhr.plot <- ggplot(cleveland, mapping = aes(x = max_hr, y = heart_disease)) +
  geom_boxplot() +
  labs(x = "Maximum Heart Rate (bpm)", y = "Heart Disease") +
  theme(axis.text.x = element_text(size = 12), axis.title.x = element_text(size = 12), 
        axis.title.y = element_text(size = 12), axis.text.y = element_text(size = 12))

bxplots <- grid.arrange(trestbps.plot, chol.plot, maxhr.plot, nrow=2)
```

![](heartdisease_files/figure-gfm/bxplots-1.png)<!-- -->

``` r
bxplots
```

    TableGrob (2 x 2) "arrange": 3 grobs
      z     cells    name           grob
    1 1 (1-1,1-1) arrange gtable[layout]
    2 2 (1-1,2-2) arrange gtable[layout]
    3 3 (2-2,1-1) arrange gtable[layout]

The box plots for resting blood pressure (top left) appear similar for
the presence and absence of heart disease; the boxes (middle 50% of
data) overlap and have similar centers and boundaries. These similar
distributions suggest that there is likely not a strong relationship
between resting blood pressure and heart disease. No strong relationship
between serum cholesterol and heart disease is visible either, for these
distributions are close in center and spread (top right). However, the
last box plot does show a relationship between maximum heart rate
achieved during exercise and heart disease. The trend of maximum heart
rate appears to be higher for the absence of heart disease compared to
the presence of heart disease, and there is little overlap between the
two boxes.

### Split the data into training set (75%) and testing set (25%)

Now that we are familiar with the trends in our data, we can begin the
machine learning process. The dataset is randomly split into the
training set and the testing set, with 75% in the training set (228
observations) and 25% in the test set (75 observations). We will build
our models on the training set and evaluate their performance on the
test set.

``` r
heart.split <- initial_split(cleveland)
heart.train <- training(heart.split)
heart.test <- testing(heart.split)
```

### Logistic Regression model using all 8 predictors

Logistic Regression is a modeling technique commonly used in the
biological sciences for categorical outcomes. Logistic Regression
computes the probability of a discrete event and classifies each
observation based on this probability. We will begin by building a
logistic regression model using all 8 predictor variables.

``` r
heart.full <- glm(heart_disease~., data = heart.train, family = "binomial")
summary(heart.full)
```


    Call:
    glm(formula = heart_disease ~ ., family = "binomial", data = heart.train)

    Deviance Residuals: 
       Min      1Q  Median      3Q     Max  
    -2.631  -0.693  -0.204   0.683   2.398  

    Coefficients:
                       Estimate Std. Error z value Pr(>|z|)    
    (Intercept)        -2.71187    2.49333   -1.09  0.27675    
    age                 0.02628    0.02430    1.08  0.27948    
    sexmale             1.66739    0.43478    3.84  0.00013 ***
    cpatypical angina  -1.53173    0.51700   -2.96  0.00305 ** 
    cpnon-anginal pain -2.00644    0.46227   -4.34  1.4e-05 ***
    cptypical angina   -1.75341    0.65819   -2.66  0.00772 ** 
    trestbps            0.02104    0.01041    2.02  0.04326 *  
    chol                0.00501    0.00349    1.44  0.15121    
    fbslow              0.15676    0.50538    0.31  0.75642    
    max_hr             -0.02383    0.00981   -2.43  0.01510 *  
    exercise_anginayes  1.12680    0.43292    2.60  0.00925 ** 
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    (Dispersion parameter for binomial family taken to be 1)

        Null deviance: 315.64  on 227  degrees of freedom
    Residual deviance: 203.76  on 217  degrees of freedom
    AIC: 225.8

    Number of Fisher Scoring iterations: 5

The variables found to be significant by the logistic regression model
are: sex, chest pain type, resting blood pressure, maximum heart rate
reached during exercise, and exercise induced angina. Age, serum
cholesterol, and fasting blood sugar were found to be insignificant
predictors of heart disease. This result was expected from exploratory
analysis of fasting blood sugar and cholesterol but unexpected from
exploratory analysis of age.

### Logistic regression model with age, fasting blood sugar, and cholesterol removed

We can set up another logistic regression model using an engine, recipe,
and work flow. The engine will be set for logistic regression and the
recipe will remove age, cholesterol, and fasting blood sugar from our
predictors. The work flow will allow the model to fit the training data
following the recipe we specified.

``` r
# set engine
heart_model <- logistic_reg() %>%
  set_engine("glm")

# create recipe
heart_recipe <- recipe(heart_disease ~., data = heart.train) %>%
  step_rm(fbs) %>%
  step_rm(age) %>%
  step_rm(chol) %>%
  step_zv(all_predictors())

# build work flow
heart_wflow <- workflow() %>%
  add_model(heart_model) %>%
  add_recipe(heart_recipe)

# fit training data through the work flow 
heart_fit <- heart_wflow %>%
  fit(data = heart.train)
tidy(heart_fit)
```

    # A tibble: 8 x 5
      term               estimate std.error statistic    p.value
      <chr>                 <dbl>     <dbl>     <dbl>      <dbl>
    1 (Intercept)          0.105    1.78       0.0589 0.953     
    2 sexmale              1.45     0.407      3.57   0.000362  
    3 cpatypical angina   -1.57     0.506     -3.10   0.00196   
    4 cpnon-anginal pain  -2.08     0.455     -4.57   0.00000480
    5 cptypical angina    -1.72     0.645     -2.67   0.00748   
    6 trestbps             0.0251   0.00984    2.55   0.0106    
    7 max_hr              -0.0262   0.00876   -2.99   0.00276   
    8 exercise_anginayes   1.12     0.423      2.65   0.00807   

The above table provides summary statistics for our logistic regression
model. The far right column shows the p-value for each predictor, and
p-values below 0.05 indicate a significant variable. All p-values are
less than 0.05, so we can infer all variables are adding predictive
power to our model (fasting blood sugar, cholesterol, and age had
p-values greater than 0.05 in the previous model, hence their removal).
The signs of the estimates calculated for each variable provide insights
into the relationship each predictor has with heart disease. Estimates
that are positive indicate a higher probability of heart disease and
estimates that are negative indicate a lower probability of heart
disease for someone with that condition. The estimate for male sex is
1.45, so we can conclude that male patients are more likely than female
patients to develop heart disease if all other variables are held
constant. Similarly, the estimate for exercise induced angina is 1.12,
so a patient with exercise induced angina is more likely than a patient
without exercise induced angina to develop heart disease provided all
else is equal. The estimate for resting blood pressure is 0.0251, which
indicate that as resting blood pressure increases, the likelihood of
someone having heart disease increases as well. The opposite is true for
maximum heart rate, for the negative estimate of -0.0262 suggests that
the higher someone can get their heart rate during exercise the lower
their chances of having heart disease. The estimates for chest pain type
are slightly more complicated, for this variable is not binary or
numeric but instead a factor with 4 levels. The baseline level for chest
pain type is asymptomatic, and the estimates for the other 3 levels must
be interpreted relative to the baseline level. The estimates for
atypical angina chest pain, non-angina chest pain, and typical angina
chest pain are all negative which suggests that the probability of
someone having heart disease is lower for someone with atypical angina
chest pain, non-angina chest pain, or typical angina chest pain compared
to someone who is asymptomatic. This pattern of chest pain type is not
in accordance with published symptoms of heart disease, so may need
further investigation.

### Receiver Operating Characteritic Technique (ROC)

The predictive power of the logistic regression model can be tested
using the receiver operating characteristic (ROC) technique. This
technique determines the diagnostic ability and accuracy rate of the
model. ROC plots the model’s sensitivity (ability to correctly predict
people with heart disease as having heart disease, equivalent to the
true positive rate) versus specificity (ability to correctly predict
people without heart disease as not having heart disease, equivalent to
the true negative rate). The sensitivity is plotted on the y-axis and
the specificity is plotted on the x-axis for varying threshold values
between 0 and 1. The threshold value determines the cutoff point for
predicting 1 (indicating presence of heart disease) versus 0 (indicating
absence of heart disease). The optimal threshold value maximizes both
sensitivity and specificity, for this is equivalent to maximizing true
positive rate and true negative rate. The area under the curve (AUC) of
the sensitivity-specificity graph measures the model’s ability to
distinguish between classes. The computed AUC value reflects our model’s
binary classification ability and the returned value is between 0 and 1,
with a perfect model having an AUC score of 1.

``` r
heart.train.pred = predict(heart_fit, new_data = heart.train)

traincomp <- data.frame(heart.train$heart_disease, heart.train.pred)
colnames(traincomp) <- c("train.response", "train.prediction")
traincomp <- traincomp %>%
  mutate(train.response = factor(case_when(train.response == "absence" ~ 0,
                                    train.response == "presence" ~ 1))) %>%
  mutate(train.prediction = factor(case_when(train.prediction == "absence" ~ 0,
                                    train.prediction == "presence" ~ 1)))

heart.roc <- roc(response = ordered(traincomp$train.response), 
                 predictor = ordered(traincomp$train.prediction))

rocplot <- plot(heart.roc, print.thres = "best", main = "Receiver Operating Characteritic Technique Plot")
```

![](heartdisease_files/figure-gfm/rocplot-1.png)<!-- -->

``` r
rocplot
```


    Call:
    roc.default(response = ordered(traincomp$train.response), predictor = ordered(traincomp$train.prediction))

    Data: ordered(traincomp$train.prediction) in 119 controls (ordered(traincomp$train.response) 0) < 109 cases (ordered(traincomp$train.response) 1).
    Area under the curve: 0.784

``` r
print(auc(heart.roc))
```

    Area under the curve: 0.784

Our model for prediction of heart disease has an AUC value of 0.784;
thus, the model will correctly predict a heart disease diagnosis from a
negative diagnosis 78.4% of the time, given new data. The optimal
threshold for the model is 0.500, which means observations with
predicted probabilities \< 0.500 will be classified as not having heart
disease and observations with predicted probabilities \> 0.500 will be
classified as having heart disease. The specificity at the optimal
threshold is 0.815, which corresponds to a 81.5% true negative rate and
18.5% false positive rate. The sensitivity at the optimal threshold is
0.752, which corresponds to a 75.2% true positive rate and 24.8% false
negative rate.

### Perform 5-fold cross validation

We will next perform 5 fold cross validation, which is a re-sampling
procedure that estimates the predictive skill of a model to classify
unseen data. First the training dataset is randomly shuffled, then split
into 5 groups (called folds). 1 fold is held out as an impromptu test
set, and the other 4 folds are joined as an impromptu training set. A
logistic regression model is trained on the 4 fold training set. The
accuracy of this model is evaluated on the 1 fold held out and this
accuracy score is stored. This model is then discarded and the impromptu
test set shifts to a fold not yet held out. The impromptu train set
shifts to the other 4 folds not in the impromptu test set and another
model is built and evaluated. This process is repeated for a total of 5
times and so there are 5 model evaluation scores (Brownlee).

``` r
set.seed(470)
folds <- vfold_cv(heart.train, v=5)

heart_fit_rs <- heart_wflow %>%
  fit_resamples(folds)

metrics <- data.frame(collect_metrics(heart_fit_rs, summarize = FALSE))

metrics <- metrics %>%
  select(-.config)
colnames(metrics) <- c("Fold", "Metric", "Estimator", "Estimate")
metrics
```

        Fold   Metric Estimator Estimate
    1  Fold1 accuracy    binary   0.8043
    2  Fold1  roc_auc    binary   0.9371
    3  Fold2 accuracy    binary   0.7826
    4  Fold2  roc_auc    binary   0.8788
    5  Fold3 accuracy    binary   0.7391
    6  Fold3  roc_auc    binary   0.8135
    7  Fold4 accuracy    binary   0.7333
    8  Fold4  roc_auc    binary   0.8043
    9  Fold5 accuracy    binary   0.8000
    10 Fold5  roc_auc    binary   0.8267

The accuracy rate and AUC are computed and shown for each of the five
folds. The accuracy of each fold reflects the corresponding model’s
proportion of correctly classified individuals in the impromptu test set
(the held out fold). The AUC scores reflect each models’ probability to
correctly classify individuals as having heart disease or not. The
accuracy of each model ranges between 0.7333 (fold 4) and 0.8043 (fold
1). The AUC scores range between 0.8043 (fold 4) and 0.9371 (fold 1). We
can expect our model to have an accuracy rate in this range when we
apply it to the test set.

### Generate predictions on testing data

We have now evaluated our model’s diagnostic ability and predictive
power and are ready to use it on the test set. The accuracy,
specificity, and sensitivity are all measured and printed.

``` r
heart_disease_pred <- predict(heart_fit, new_data = heart.test) %>%
  bind_cols(heart.test %>% select(heart_disease))

test_accuracy <- accuracy(heart_disease_pred, truth = heart_disease, estimate = .pred_class)
test_specificity <- spec(heart_disease_pred, truth = heart_disease, estimate = .pred_class)
test_sensitivity <- sens(heart_disease_pred, truth = heart_disease, estimate = .pred_class)

test.values <- data.frame(test_accuracy$.estimate, 
                          test_sensitivity$.estimate, 
                          test_specificity$.estimate)
colnames(test.values) <- c("Test set Accuracy", "Test set Sensitivity", "Test set Specificity")
test.values
```

      Test set Accuracy Test set Sensitivity Test set Specificity
    1               0.8               0.8444               0.7333

The model has a test set accuracy of 0.80, which indicates 80% of the
patients in the test set are correctly classified as having heart
disease or not. The true positive rate on the test set is 84.44% and the
true negative rate is 73.33%. The high true positive rate of our model
is promising, and indicates a high predictive ability of our model to
correctly classify individuals that have heart disease. The high true
negative rate of our model corresponds to a low false positive rate,
which means individuals without heart disease will largely be classified
as not having heart disease. Therefore, this model may be used to
accurately diagnose people with heart disease based of their
demographics and risk factors while avoiding misdiagnosis, which would
be inconvenient and cause unnecessary stress for people without heart
disease.

### Conclusion

Heart disease is a deadly condition that affects people across the
world. Certain risk factors and demographics may determine how
susceptible a person is to developing heart disease, and the resulting
model may be used to determine if someone has heart disease or not based
on these factors. Some of the predictors used in this model are not
controllable, but others may be used to dictate a person’s lifestyle
choices. The logistic regression model proved to be effective, given the
accuracy rate of 0.80 on the test set. Sex, chest pain type, resting
blood pressure, maximum heart rate reached during exercise, and exercise
induced angina were found to be significant predictors of heart disease.
However, the distribution of chest pain type was not in accordance with
published symptoms of heart disease, so further investigation is needed.
Age, cholesterol, and fasting blood sugar were found to be insignificant
variables and so they were not included in the final model. An
interesting possibility for further analysis could examine the
relationship between gender and symptoms of heart disease, for there is
differing information regarding its duration and intensity of symptoms
for males and females. This model showed a substantial influence of sex
on the prediction of heart disease, so it may be interesting to
investigate an interaction between symptoms and sex for sufferers of
heart disease.

### References

Brownlee, Jason. “A Gentle Introduction to k-Fold Cross-Validation.”
Machine Learning Mastery, 2 Aug. 2020,
<https://machinelearningmastery.com/k-fold-cross-validation/>.

“Heart Disease.” Centers for Disease Control and Prevention, U. S.
Department of Health and Human Services, 19 Jan. 2021,
<https://www.cdc.gov/heartdisease/index.htm>.

### Data Set Source

Dataset source by UCI Machine Learning Repository:
<https://archive.ics.uci.edu/ml/datasets/Heart+Disease>

Authors: 1. Hungarian Institute of Cardiology. Budapest: Andras Janosi,
M.D.

2.  University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.

3.  University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.

4.  V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:
    Robert Detrano, M.D., Ph.D.

Donor: David W. Aha (aha ‘@’ ics.uci.edu) (714) 856-8779
