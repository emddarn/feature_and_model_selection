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

###	Feature Selection Techniques
Three different approaches were considered for feature selection: correlation-based, feature permutation, and mutual information; as described below.

**Correlation-based:** Pearson correlation coefficient is computed to compute the relation among the features. The “mpg” parameter illustrated strong correlation (>0.65) with most of the features, the top three being “wt”, “disp”, and “cyl” (>0.85).
However, Pearson correlation coefficient is not a good indicator of non-linear relationships. For this reason, I have employed other approaches to identify any non-linear relation.

**K-best selection:** This approach selects k features according to the highest scores, calculated based on Mutual Information (MI). Higher MI values indicate stronger dependency between two random variables, with 0 indicating independent variables. With this method, three top most significant features are also “wt”, “disp”, and “cyl”.

**Feature permutation:** Permutation importance is calculated after a model has been fit. It works based on the principle that randomly re-ordering a single feature should provide less accurate predictions, if the model heavily relies on that feature for prediction. The top three features using this approach are “wt”, “disp”, and “hp”.

<p align="center">
  <img src="report_diagrams/Fig_1_relationship.jpg?raw=true" alt="fig_1"/>  <br/>
  Fig 1: Relationship among mpg and the selected features
</p> 

Two approaches (correlation-based, and k-best) indicated “wt”, “disp”, and “cyl” to be the top features in predicting “mpg”; thus, for these features were selected for modelling as part of this activity and reporting. Fig 1 illustrates the strong correlation of these features with the “mpg” variable. 

##	Predictive Modelling

### Algorithm Selection and Evaluation
The goal of this activity is to build a model to predict the parameter “mpg” for any given set of selected parameters mentioned in previous section. This being a regression problem, there are several regression algorithms suitable for this purpose. For this activity, both linear and non-linear algorithms, such as Linear Regression, Decision Tree, and Random Forest based regression algorithms, were applied.

For algorithm evaluation, Mean Square Error (MSE) metric was used for comparison. As the training sample size was small, the K-fold cross-validation (k=5) approach was applied. In addition, the training data was split manually to compare the training and validation MSE for further insight.

<p align="center">
  <img src="report_diagrams/Fig_2_mse_comparison.jpg?raw=true" alt="fig_2"/>  <br/>
  Fig 2. Average MSE comparison using K-fold cross-validation
</p> 

The Decision Tree approach appears to be over-fitting the training data (MSE = 0) and showed very high (MSE = 9.67) for the validation data (please refer to the attached CancerNSW_Analysis.html). The fitting for Linear Regression was erroneous and showed lower MSE (= 3.93) for the validation set compared to training data (7.25). Reviewing the data splitting method, adding more features may improve the model performance. 

Considering the performance of the three models, Random Forest regression algorithm provides the reasonably well performance (mean MSE = 2.81 using cross-validation), thus, was selected to build the final model as part of this exercise.

###	Algorithm Fine Tuning and Final Model
The next step was to fine-tune the selected Random Forest Regression algorithm for this dataset. For this exercise, parameters related to number of trees in the forest, maximum leaf nodes (to regularise the model), whether to use bagging or not were evaluated through K-fold cross-validation of the training dataset. The most accurate combination of parameters for this case found to be, 3 for number of trees and 4 for maximum leaf nodes, with no bootstrap aggregation; which were selected for the final model.

<center>
<table>
<thead>
  <tr>
    <th rowspan="2">Cross-validation</th>
    <th colspan="2">Mean Square Error</th>
  </tr>
  <tr>
    <td>Mean</td>
    <td>Standard Deviation</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Yes</td>
    <td>2.74</td>
    <td>0.79</td>
  </tr>
  <tr>
    <td>No</td>
    <td>2.67</td>
    <td>-</td>
  </tr>
</tbody>
</table>  
Table 1. MSE measure of the final model in predicting “mpg”  
</center>

<br/><br/>
Since the amount of test data is not large enough, the final model applied 3-fold cross validation which showed an average MSE of 2.74 (standard deviation of 0.79). This means that while predicting the value of mpg, on an average the predicted “mpg” could be as close as 1.95 (=2.74-0.79) or as far as 3.58 (=2.79+0.79) from the actual value.

<p align="center">
  <img src="report_diagrams/Fig_3_feature_importance.jpg?raw=true" alt="fig_3"/>  <br/>
  Fig 3. Importance of feature in predicting mpg
</p>  


From feature importance analysis of the model in Fig. 3, it is found that weight of the car is the number one predictor for miles-per-gallon (mpg) estimation.

##	Recommendation
The feature selection, feature engineering are iterative process; further analysis is required to improve the model performance. 
For the sake of simplicity in this exercise, random sampling was used to separate training and test dataset. However, for small dataset, stratified sampling should be used to have sufficient number of instances for each stratum in training and test set. Since “wt” is a strong predictor for “mpg” (Fig 3), stratified sampling using “wt” might provide better training and test set, which might lower the generalisation error for any new data. 

Additionally, other learning algorithms like support vector machines based regressor can also be tested; however, for that case feature scaling is a must.

## Appendix
* Fig 4: Histogram Plot for each feature (for detail please refer to the Regression_Analysis.ipynb)  
<p align="center">
  <img src="report_diagrams/Fig_4_histogram.jpg?raw=true" alt="fig_4"/>

* Fig 5: Bivariate relation analysis using scatter plot (refer to the Regression_Analysis.ipynb)  
<p align="center">
  <img src="report_diagrams/Fig_5_bivariate_analysis.jpg?raw=true" alt="fig_5"/>
</p>




