# Predictive Analysis of MtCars Dataset
## Executive Summary
The goal of this activity is to build a machine learning model to predict the mile/gallon (mpg) variable using three or less predictor features using MtCars dataset. The MtCars dataset has 12 different columns describing the fuel consumption (mpg), car model, and 10 other design and performance metrics for 32 automobiles. To build a predictive model, at first the available data was split into training and test set. The training dataset was analysed for data insight and pattern identification, where new features were included from the insights. For feature selection, three different approaches were adopted to identify three most important features for mpg prediction: “wt”, “disp”, and “cyl”. Both linear and non-linear learning algorithm were compared to identify the Random Forest Regressor as base algorithm, which was then fine-tuned using grid-search technique. Finally the fine-tuned model is employed to calculate the benchmark performance of the developed model for any unseen data.
## Dataset
### Source Data
The MtCars dataset has 12 different columns - all of which have 32 entries and no null values, meaning that no feature is missing for any of the entries. All attributes are numerical, except the “car” field, which has text attribute. 
Since no data is missing in this dataset, data imputation was not required. 
The data range scales among the features are not considerably disperse (refer to Appendix Fig. 3). Moreover, for demonstration purpose in this activity, performance comparison was carried out among tree, linear regression, and ensemble based algorithms. These algorithms are not heavily affected by the scales. For that reason, feature scaling or any other feature transformation activities were not adopted to maintain simplicity.
As the count of each car model is 1, this feature does not carry any significant information relating the target “mpg” variable. Hence, this feature was discarded from the analysis.
###	Train-test Dataset
It is very important not to consider the test set data while selecting features and algorithms to avoid the “data snooping bias”. If we find any interesting pattern from the whole dataset and use that information to select features or algorithms, the model might work well with test set data, but will most likely to fail to generalise well for any new input. For that reason, the test set data (test_set variable) was separated at the very beginning and used only for final model validation.
For this exercise, random sampling was adopted for the sake of simplicity; even though it generally works well for large dataset. For a small dataset like MtCars, stratified sampling based on most significant attribute is a better choice.
##	Feature Selection
###	Training Data Insight
Different visualisation techniques were used to gather data insight from the training dataset. It can be observed from the histogram plot (Appendix Fig. 4) that each feature attribute is not heavily skewed. The data are to somewhat distributed over the whole range.
From the scatter plots (Appendix Fig. 5) of each feature-pair, we can see that “mpg” has strong linear correlation with “cyl”, “disp”, “wt”. The relations of “mpg” with “hp”, “drat” are also evident.
The strong negative correlation (0.899) between “wt” and “mpg” is in line with our experience. Cars with low weights are more likely to have higher mpg. The “disp” feature is strongly related with “hp” and “wt”.
###	Feature Engineering
Since “hp” and “wt” have strong correlation with “disp” attribute (0.795898 and 0.903760 respectively), these features were combined to compute new features “hp_per_disp” and “wt_per_disp” features. Another feature “qsec_per_drat” was added to identify if these combinations d any significant impact on the “mpg” prediction.
However, in the feature selection process, described below, it was found that they did not contribute significantly while predicting “mpg”.
