# blog:https://github.com/aarshayj/Analytics_Vidhya/blob/master/Articles/Parameter_Tuning_GBM_with_Example/GBM%20model.ipynb
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

import matplotlib.pylab as plt
#%matplotlib inline

"""
GBM Discussion
The data here is taken form the Data Hackathon3.x - http://datahack.analyticsvidhya.com/contest/data-hackathon-3x

Load Data:
The data has gone through following pre-processing:

City variable dropped because of too many categories
DOB converted to Age | DOB dropped
EMI_Loan_Submitted_Missing created which is 1 if EMI_Loan_Submitted was missing else 0 | EMI_Loan_Submitted dropped
EmployerName dropped because of too many categories
Existing_EMI imputed with 0 (median) - 111 values were missing
Interest_Rate_Missing created which is 1 if Interest_Rate was missing else 0 | Interest_Rate dropped
Lead_Creation_Date dropped because made little intuitive impact on outcome
Loan_Amount_Applied, Loan_Tenure_Applied imputed with missing
Loan_Amount_Submitted_Missing created which is 1 if Loan_Amount_Submitted was missing else 0 | Loan_Amount_Submitted dropped
Loan_Tenure_Submitted_Missing created which is 1 if Loan_Tenure_Submitted was missing else 0 | Loan_Tenure_Submitted dropped
LoggedIn, Salary_Account removed
Processing_Fee_Missing created which is 1 if Processing_Fee was missing else 0 | Processing_Fee dropped
Source - top 2 kept as is and all others combined into different category
Numerical and One-Hot-Coding performed

"""

train = pd.read_csv('data/train_modified.csv')
target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'
# 查看正负样例的比
train['Disbursed'].value_counts()

"""
Define a function for modeling and cross-validation
This function will do the following:

fit the model
determine training accuracy
determine training AUC
determine testing AUC
perform CV is performCV is True
plot Feature Importance if printFeatureImportance is True
"""

def modelfit(alg, dtrain, dtest, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(estimator=alg,
                                                    X=dtrain[predictors],
                                                    y=dtrain['Disbursed'],
                                                    cv=cv_folds,
                                                    scoring='roc_auc')

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()

"""
Baseline Model
Since here the criteria is AUC, simply predicting the most prominent class would give an AUC of 0.5 always. Another way of getting a baseline model is to use the algorithm without tuning, i.e. with default parameters.
"""

#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, train, None, predictors)

"""
GBM Models:
There 2 types of parameters here:

Tree-specific parameters
min_samples_split
min_samples_leaf
max_depth
min_leaf_nodes
max_features
loss function
Boosting specific paramters
n_estimators
learning_rate
subsample
Approach for tackling the problem
Decide a relatively higher value for learning rate and tune the number of estimators requried for that.
Tune the tree specific parameters for that learning rate
Tune subsample
Lower learning rate as much as possible computationally and increase the number of estimators accordingly.
Step 1- Find the number of estimators for a high learning rate
We will use the following benchmarks for parameters:

min_samples_split = 500 : ~0.5-1% of total values. Since this is imbalanced class problem, we'll take small value
min_samples_leaf = 50 : Just using for preventing overfitting. will be tuned later.
max_depth = 8 : since high number of observations and predictors, choose relatively high value
max_features = 'sqrt' : general thumbrule to start with
subsample = 0.8 : typically used value (will be tuned later)
0.1 is assumed to be a good learning rate to start with. Let's try to find the optimum number of estimators requried for this.
"""

#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
param_test1 = {'n_estimators':np.array(range(20,81,10))} #学习器的个数

gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,
                                  min_samples_leaf=50,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
                       param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])

print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)


"""
So we got 60 as the optimal estimators for the 0.1 learning rate. Note that 60 is a reasonable value and can be used as it is. But it might not be the same in all cases. Other situations:

If the value is around 20, you might want to try lowering the learning rate to 0.05 and re-run grid search
If the values are too high ~100, tuning the other parameters will take long time and you can try a higher learning rate
Step 2- Tune tree-specific parameters
Now, lets move onto tuning the tree parameters. We will do this in 2 stages:

Tune max_depth and num_samples_split
Tune min_samples_leaf
Tune max_features
"""

#Grid seach on subsample and max_features
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,
                                                max_features='sqrt', subsample=0.8, random_state=10),
                       param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)

"""
Since we reached the maximum of min_sales_split, we should check higher values as well. Also, we can tune min_samples_leaf with it now as max_depth is fixed. One might argue that max depth might change for higher value but if you observe the output closely, a max_depth of 9 had a better model for most of cases. So lets perform a grid search on them:
"""

#Grid seach on subsample and max_features
param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,
                                                    max_features='sqrt', subsample=0.8, random_state=10),
                       param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])

print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_ )

modelfit(gsearch3.best_estimator_, train, None, predictors)


"""
Tune max_features:
"""
#Grid seach on subsample and max_features
param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,
                            min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10),
                       param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])

print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)

"""
Step3- Tune Subsample and Lower Learning Rate

"""
#Grid seach on subsample and max_features
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,
                            min_samples_split=1200, min_samples_leaf=60, subsample=0.8, random_state=10, max_features=7),
                       param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])
print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)


"""
With all tuned lets try reducing the learning rate and proportionally increasing the number of estimators to get more robust results:
"""

#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=9, min_samples_split=1200,
                                         min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
modelfit(gbm_tuned_1, train, None, predictors)

# 1/10th learning rate
#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm_tuned_2 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600,max_depth=9, min_samples_split=1200,
                                         min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)
modelfit(gbm_tuned_2, train, None, predictors)

# 1/50th learning rate
#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm_tuned_3 = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1200,max_depth=9, min_samples_split=1200,
                                         min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7,
                                         warm_start=True)
modelfit(gbm_tuned_3, train, None, predictors, performCV=False)

#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm_tuned_4 = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1500,max_depth=9, min_samples_split=1200,
                                         min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7,
                                         warm_start=True)
modelfit(gbm_tuned_4, train, None, predictors, performCV=False)





