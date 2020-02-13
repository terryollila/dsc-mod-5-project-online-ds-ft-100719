
## Final Project Submission

Please fill out:
* Student name: Terry Ollila
* Student pace: full time
* Scheduled project review date/time: 2/11/2019 12pm
* Instructor name: James Irving
* Blog post URL: https://terryollila.github.io/repetitive_model_fitting_in_classification


# Premise

This analysis will explore a dataset from the 2017 National Health Interview Survey (NHIS) provided to the public by the Center for Disease Control and Prevention (CDC). The purpose is to create a means of predicting mental illness of debilitating depression and anxity based on numerous societal, personal, and economic factors as submitted by the American public to the CDC.

The survey itself consists of responses from 26,000 individual adults, who are the only targets in scope for this report. Only a fraction of these records will be used as classifying something as granular as mental illness will necessetate a paring down of the original data set.

The target variable in this report is an adult individual's diagnose depression and anxiety as classified as mentall illness severe enough to imare activities.

This report is not intended to be used as a guide for avoiding or relieving mental illness, but as a tool to identify potential warning signs and to investigate societal, economic, and lifestyle factors.

# Libraries and Functions

## Libraries


```python
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.svm import SVC
from pydotplus import graph_from_dot_data
from xgboost import XGBClassifier
from IPython.display import Image  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
random.seed(42)
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
```


```python
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)
plt.style.use('ggplot')
```

## Functions

### Tree Builder

The primary tool that will take a classifier and data and create the model.


```python
def tree_builder(data, target, max_depth=None, 
                 classifier=DecisionTreeClassifier, params=None):
    """Sets up train/test splits, undersampling, fitting, output, testing
    results, and graphs for multiple types of classifiers.
    
    Arguments:
        
        data (DataFrame):
            Will be used to create the training and testing samples.
            
        target (Series):
            The target or y value of the data to be tested.
            
        max_depth (int):
            Sets the max_depth parameter for the classifier, if it has one.
            
        classifier (classifier):
            The classifier to be used. Can be decision tree, random forest, 
            KNN, XGBoost, or SVC.
            
        params (dictionary):
            Dictionary of any additional classifier parameters to be included
            for modeling.
            
    Returns: 
        If decisioni tree or random forest generator, returns a series of 
        most important features.""" 
      
    # Setting up train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(data, 
                                                    target,
                                                    random_state=42,
                                                    test_size=.25)
    # Need to under-sample due to small target set.
    rus = RandomUnderSampler(random_state=42)
    Xt_resampled, yt_resampled = rus.fit_resample(X_train, y_train)
    
    # Fitting model and creating predictions etc.
    apple_tree = classifier(**params)
    apple_tree.fit(Xt_resampled, yt_resampled)
    y_train_pred = apple_tree.predict(X_train)
    y_test_pred = apple_tree.predict(X_test)
    
    # Not all classifiers have decision functions, so the following will catch
    # those that do and the others will get a different roc_auc value.
    try:
        y_score = apple_tree.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
    except:
        roc_auc = roc_auc_score(y_test, y_test_pred)

    # Printing up all the scores.
    print('Train Accuracy: ', accuracy_score(y_train, y_train_pred))
    print('Test Accuracy: ', accuracy_score(y_test, y_test_pred))
    print('Precision: ', precision_score(y_test, y_test_pred))
    print('Recall: ', recall_score(y_test, y_test_pred))
    print('f1_score: ', f1_score(y_test, y_test_pred))
    
    # Check the AUC for predictions
    print('\nAUC is :{0}'.format(round(roc_auc, 2)))

    # Create and plot a confusion matrix 
    so_confused = confusion_matrix(y_test, y_test_pred)    
    
    # Using the MLExtend library since the sklearn version kept coming up 
    # with invisible numbers due to the color scheme.
    plot_confusion_matrix(so_confused, figsize=(7,7), colorbar=True,
                          show_normed=True, cmap=plt.cm.Greens)
    plt.tick_params(labelsize=20)
    plt.title('Identification of Depression/Anxiety')
    plt.show();
    
    # Plotting an importance bar graph for those classifiers that support it.
    if classifier != KNeighborsClassifier and classifier != SVC:
        print(pd.Series(apple_tree.feature_importances_,
                  index=X_train.columns).sort_values(ascending=False).head(10))
        df_importance = pd.Series(apple_tree.feature_importances_, 
                                  index=X_train.columns)
        df_imp_export = df_importance.sort_values(ascending=False)
        df_importance = df_importance.sort_values(ascending=True).tail(10)
        df_importance.plot(kind='barh', figsize=(8,10))
        plt.title('Most Important Features')
        plt.ylabel('Feature Name')
        plt.xlabel('Feature Importance')
        plt.show()
    
    # Plotting an ROC/AUC graph for those that support it.
    try: 
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    except:
        pass
    
    # Creating a decision tree classifier tree graph because it's cool.
    if classifier == DecisionTreeClassifier:
        dot_data = export_graphviz(apple_tree, out_file=None, 
                                   feature_names=data.columns, 
                                   class_names=np.unique(target).astype('str'), 
                                   filled=True, rounded=True, 
                                   special_characters=True)

        # Draw graph
        graph = graph_from_dot_data(dot_data)  
        # Show graph
        Image(graph.create_png()) 
        return Image(graph.create_png()) 
    
    # If supported by the classifier, return the feature importances.
    if classifier != KNeighborsClassifier and classifier != SVC:
        return df_imp_export
```

### Grid Search

Used to find best parameters for modeling.


```python
def grid_search(clf, data, target, params, verbose, n_estimators=100,
                scoring='accuracy'):
    """Samples data and then runs it through the sklearn grid search to
    determine optimal parameters for various classifiers.
    
    Parameters:

        classifier (classifier object):
            The classifier to be used. Can be decision tree, random forest, 
            KNN, XGBoost, or SVC.
            
        data (DataFrame):
            Will be used to create the training and testing samples.
            
        target (Series):
            The target or y value of the data to be tested.
            
        params (dictionary):
            Dictionary of any additional classifier parameters to be included
            for modeling.
            
        n_estimators (int):
            Number of instances to use in a random forest generator.
            
        scoring (string):
            The scoring type to be maximized for whatever classifire is 
            indicated.
            
        returns: 
            A dictionary of parameters deemed by the grid search to be optimum.
            """
    
    X_train, X_test, y_train, y_test = train_test_split(data, 
                                                    target,
                                                    random_state=42,
                                                    test_size=.25)
    
    # Need to under-sample due to small target set.
    rus = RandomUnderSampler(random_state=42)
    Xt_resampled, yt_resampled = rus.fit_resample(X_train, y_train)
    
    # Setting up the model variable for however it winds up being used.
    mod = None
    
    # If the classifier accepts a n_estimators parameter, I'll use it.
    if clf == SVC or clf == KNeighborsClassifier or clf == DecisionTreeClassifier:
        mod = clf()
    else:
        mod = clf(n_estimators=n_estimators)
        
    #Running the grid search.
    grid_search = GridSearchCV(mod, cv=5, param_grid=params,
                               return_train_score=True, verbose=verbose,
                               scoring=scoring)
    grid_search.fit(Xt_resampled, yt_resampled)
    
    # Create mean training and test scores.
    training_score = np.mean(grid_search.cv_results_['mean_train_score'])
    testing_score = grid_search.score(X_test, y_test)
    
    # Spit out the results.
    print(f"Mean Training Score: {training_score :.2%}")
    print(f"Mean Test Score: {testing_score :.2%}")
    print("Best Parameter Combination Found During Grid Search:")
    print(grid_search.best_params_)
    
    # Return the parameters so they can be seen and contemplated.
    return grid_search.best_params_
```

### Find Important Features

Used in the initial exploration to establish most important features for initial modeling.


```python
def find_important_features(data, target, max_depth=None, 
                 classifier=DecisionTreeClassifier, params=None):
    """Stripped down version of modeling function for data exploration to find
    most important features prior to running the data through the rest of the 
    models.
    
    Parameters:
    
        data: DataFrame
            Feature set with one-hot encoding (if needed).
        
        target: Series
            Target data for model to be trained against.
            
        max_depth: int
            Value for how deep to go with tree levels, typically returned by
            grid search for best value.
            
        classifier: classifier object 
            Type of classifier to use. Will generally be random tree forest
            since that's the one that returns most important features.
            
        params: dict
            Dictionary with keys of parameter name and values of parameter
            values.
            
        Returns:
            DataFrame of features ordered from most to least important."""
    
    X_train, X_test, y_train, y_test = train_test_split(data, 
                                                    target,
                                                    random_state=42,
                                                    test_size=.25)
    
    # Setting up train and test sets.
    rus = RandomUnderSampler(random_state=42)
    Xt_resampled, yt_resampled = rus.fit_resample(X_train, y_train)
    
    # Setting up model and fitting.
    apple_tree = classifier(**params)
    apple_tree.fit(Xt_resampled, yt_resampled)
    
    df_importance = pd.Series(apple_tree.feature_importances_, 
                              index=X_train.columns)
    df_imp_export = df_importance.sort_values(ascending=False)
    
    #Export the importances.
    return df_imp_export
```

### Cat Scatter

Plots a single feature against the target.


```python
def cat_scatter(feature, target, cut_last=False):
    """Plots a single feature against a target in a scatter graph.
    
    Parameters:
        
        feature: series
            A single column from the dataset.
            
        target: series
            The target to be predicted in the model.
            
        cut_last: bool default False
            Some series have a catchall or 'other' value as the final category.
            This cuts that from the graph.
        
        Returns: Nothing."""
        
    # Concatenate the feature and target for plotting, then group together.
    con = pd.concat([feature, target], axis=1)
    group = con.groupby(feature.name).mean()
    
    # If cutting, take the last value off for both, or just plot.
    plt.title(feature.name + ' vs. \n Depression/Anxiety')
    plt.xlabel(feature.name + ' Category (Bin)')
    plt.ylabel('Average Depression/Anxiety Value')
    if cut_last:
        x = list(group.index[:-1])
        plt.scatter(x, group.AFLHCA17[:-1]);
    else:
        plt.scatter(group.index, group.AFLHCA17);
```

# Import Data

This is the adult report from the CDC data set, plus the 'family' data, which will be merged in order to provide additional information regarding income etc. Merge will be made on the household number to align individual data with family.


```python
cdc_ad = pd.read_csv('Adult/samadult.csv')
```


```python
cdc_fam = pd.read_csv('Family/familyxx.csv')
```

Parsing out the necessary data right away to avoid excessive processing time later on.


```python
cdc_fam_reduced = cdc_fam[['HHX',
                             'FM_SIZE', 
                             'FM_KIDS', 
                             'FM_ELDR',
                             'FM_STRCP',
                             'FSRUNOUT',
                             'FHICOVYN',
                             'FTANFYN',
                             'FINTR1YN',
                             'FDIVDCT',
                             'INCGRP5',
                             'RAT_CAT5',
                             'FSNAP',]]
```

### Combining data sets

This is the bigger data set I started with in order to get the bigger picture and find out what was or was not significant.


```python
combined_df = cdc_ad[['HHX',
                        'REGION',
                        'SEX',
                        'HISPAN_I',
                        'RACERPI2',
                        'MRACRPI2',
                        'R_MARITL',
                        'PAR_STAT',
                        'EVERWRK',
                        'INDSTRN1',
                        'INDSTRN2',
                        'OCCUPN1',
                        'OCCUPN2',
                        'SUPERVIS',
                        'WRKCATA',
                        'BUSINC1A',
                        'LOCALL1B',
                        'HOURPDA',
                        'PDSICKA',
                        'ONEJOB',
                        'WRKLYR4',
                        'DBHVPAN',
                        'HRAIDNOW',
                        'VIMGLASS',
                        'SMKREG',
                        'SMKQTTP',
                        'VIGFREQW',
                        'MODTP',
                        'ALCLIFE',
                        'ALC12MWK',
                        'ALC5UPY1',
                        'ALCAMT',
                        'BMI',
                        'ASICPUSE',
                        'ASINKNT',
                        'ASISIM',
                        'ASISIF',
                        'ASINBILL',
                        'ASISLEEP',
                        'HOM_USM1',
                        'MBO_MAN1',
                        'MBO_MND1',
                        'MBO_SPR1',
                        'MBO_IMG1',
                        'MBO_PRO1',
                        'YTQU_YG1',
                        'YTQU_TA1',
                        'YTQU_QG1',
                        'AWEBUSE',
                        'AWEBOFNO',
                        'AWEBOFTP',
                        'AWEBEML',
                        'AWEBMNO',
                        'SMKNOW',
                        'AWEBMTP']].copy()
```

I then pared it down to only those features that were important in establishing predicability for the classification modeling.


```python
combined_sig_data = cdc_ad[['AFLHCA17',
                            'HHX',
                            'REGION',
                            'HISPAN_I',
                            'MRACRPI2',
                            'R_MARITL',
                            'PAR_STAT',
                            'INDSTRN1',
                            'INDSTRN2',
                            'OCCUPN1',
                            'OCCUPN2',
                            'SUPERVIS',
                            'WRKCATA',
                            'LOCALL1B',
                            'PDSICKA',
                            'ONEJOB',
                            'WRKLYR4',
                            'VIMGLASS',
                            'ALC12MWK',
                            'ASINKNT',
                            'ASISIM',
                            'ASISIF',
                            'ASINBILL',
                            'MBO_MND1',
                            'MBO_SPR1',
                            'MBO_IMG1',
                            'YTQU_YG1',
                            'AWEBEML',
                            'CIGSDAY',
                            'AWEBMTP']].copy()
```

# Clean and Transform Data 

## Convert continous values to bins

Some features were continuous but non-linear, so I put them into bins for improved processing.


```python
combined_sig_data.fillna(0, inplace=True)

combined_sig_data['AGE'] = pd.cut(cdc_ad.AGE_P, [18, 23, 33, 43, 53, 63, 73, 83],
                                  labels=[23, 33, 43, 53, 63, 73, 83])
combined_sig_data['YEARS_WORKED'] = cdc_ad.YRSWRKPA
combined_sig_data['YEARS_WORKED'].fillna(999, inplace=True)
combined_sig_data['YEARS_WORKED'] = pd.cut(combined_sig_data['YEARS_WORKED'], 
                                           bins=[-1,1,2,5,10,15,20,30,50,1000], 
                                           labels=[1,2,5,10,15,20,30,50,1000])

combined_sig_data['VIGOROUS_ACTIVITY'] = cdc_ad.VIGFREQW
combined_sig_data['VIGOROUS_ACTIVITY'] = pd.cut(cdc_ad.VIGFREQW, 
                                                bins=[-1,0,1,3,8,15,30,100],
                                                labels=[0,1,3,8,15,30,100])

combined_sig_data['MODERATE_ACTIVITY'] = cdc_ad.MODFREQW
combined_sig_data['MODERATE_ACTIVITY'] = pd.cut(cdc_ad.MODFREQW, 
                                                bins=[-1,0,1,3,8,15,30,100],
                                                labels=[0,1,3,8,15,30,100])

combined_sig_data['ALCOHOL_PER_DAY_DRANK'] = cdc_ad.ALCAMT
combined_sig_data['ALCOHOL_PER_DAY_DRANK'] = pd.cut(cdc_ad.ALCAMT, 
                                                    bins=[-1,0,1,3,5,8,12,94,100],
                                                    labels=[0,1,3,5,8,12,20,100])
combined_sig_data['ALCOHOL_PER_DAY_DRANK'].fillna(0, inplace=True)

combined_sig_data['BMI_BIN'] = cdc_ad.BMI.fillna(0)
combined_sig_data['BMI_BIN'] = pd.qcut(cdc_ad.BMI, q=5, 
                                       labels=[23,26,29,
                                               33,50])

combined_sig_data['HOURS_SLEEP'] = cdc_ad.ASISLEEP
combined_sig_data['HOURS_SLEEP'] = pd.cut(cdc_ad.ASISLEEP, bins=[0,5,6,7,8,9,
                                                                 12,15,100],
                                          labels=[5,6,7,8,9,12,15,100])

combined_sig_data['WEB_HOURS'] = cdc_ad.AWEBOFNO
combined_sig_data['WEB_HOURS'].fillna(0, inplace=True)
combined_sig_data['WEB_HOURS'] = pd.cut(combined_sig_data['WEB_HOURS'], 
                                        bins=[-1,0,1,5,10,20,40,1000],
                                        labels=[0,1,5,10,20,40,1000])
```

Taking care of null values. Because of the nature of the survey data, NaN is generally going to mean the question does not apply, so I'm making those into zeros.


```python
combined_sig_data.VIMGLASS.fillna(0, inplace=True)
combined_sig_data.ALC12MWK.fillna(0, inplace=True)
combined_sig_data.ASISIM.fillna(0, inplace=True)
combined_sig_data.ASISIF.fillna(0, inplace=True)
combined_sig_data.AWEBMTP.fillna(0, inplace=True)
```

Merging individual adult data with family data to capture additional variables such as income.


```python
cdc_combined = pd.merge(combined_sig_data, cdc_fam_reduced, how='left',
                        on='HHX')
cdc_combined.drop_duplicates(subset='HHX', keep='first', inplace=True)
```

Creating target variable: Depression and anxiety as classified as mental illness and severe enough to impare activities.


```python
test_target = cdc_combined['AFLHCA17']
test_target = test_target.fillna(0)
test_target = test_target.where(test_target == 1, 0)

# Dropping the target variable now that we have created a new series for it.
cdc_combined.drop(['AFLHCA17', 'HHX'], axis=1, inplace=True)
```

# Exploratory Data Analysis

## Categorical Investigation

Testing categorial conversions for possible non-categorical use.


```python
cat_scatter(cdc_combined['YEARS_WORKED'], test_target, cut_last=True)
```


![png](output_44_0.png)



```python
cat_scatter(cdc_combined['AGE'], test_target)
```


![png](output_45_0.png)



```python
cat_scatter(cdc_combined['VIGOROUS_ACTIVITY'], test_target, cut_last=True)
```


![png](output_46_0.png)



```python
cat_scatter(cdc_combined['MODERATE_ACTIVITY'], test_target, cut_last=True)
```


![png](output_47_0.png)



```python
cat_scatter(cdc_combined['ALCOHOL_PER_DAY_DRANK'], test_target, cut_last=True)
```


![png](output_48_0.png)



```python
cat_scatter(cdc_combined['BMI_BIN'], test_target)
```


![png](output_49_0.png)



```python
cat_scatter(cdc_combined['WEB_HOURS'], test_target, cut_last=True)
```


![png](output_50_0.png)


## Establish Best Features

Performing one-hot encoding for categorical modeling.


```python
# The existing values must first be converted into integers in order to be encoded.
cdc_combined = cdc_combined.astype(int)
you_dummies = pd.get_dummies(cdc_combined, 
                         columns=list(cdc_combined.columns),
                         drop_first=False)
```

Will first fit a random forest model in order to obtain a ranked list of most important features.


```python
# Create the parameters for the grid search to consider.
grid = {'criterion': ['gini', 'entropy'],
                'max_depth': [7,8,9,10,11],
                'min_samples_split': [2,5,10,15,20,30],
                'min_samples_leaf': [2,5,10,15,20,30],
                'max_features': [20,30,50,100]} 

# Grid search will return a set of optimal parameters for the classifier to use.
best_params_rf = grid_search(RandomForestClassifier, you_dummies, test_target, 
                          grid, verbose=False, n_estimators=100)

# Classifier returns best features to be used for the rest of the modeling,
# helping to alleviate unnecessary dimensions.
best_features_rfc = find_important_features(you_dummies, test_target, 
                                            params=best_params_rf,
                                            classifier=RandomForestClassifier)
```

    Mean Training Score: 79.58%
    Mean Test Score: 71.15%
    Best Parameter Combination Found During Grid Search:
    {'criterion': 'gini', 'max_depth': 7, 'max_features': 30, 'min_samples_leaf': 2, 'min_samples_split': 20}


Creating a correlation matrix in order to establish pos/neg metrics for the most important features.


```python
# Taking only top 100 features as defined by RFC above.
trim_dummies = you_dummies[best_features_rfc.index[:100]]
corr_dummies = pd.concat([you_dummies[best_features_rfc.index[:15]], 
                          test_target], axis = 1)

# Create a correlation matrix for determining pos/neg correlation for important
# features. Sort by original importance.
corr_dummies = corr_dummies.corr()
corr_dummies['to_sort'] = abs(corr_dummies['AFLHCA17'])
to_sort_df = corr_dummies[['AFLHCA17', 'to_sort']][:-1]
corr_dummies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ONEJOB_0</th>
      <th>FSNAP_1</th>
      <th>ONEJOB_2</th>
      <th>WRKLYR4_0</th>
      <th>INCGRP5_1</th>
      <th>FSNAP_2</th>
      <th>MBO_SPR1_1</th>
      <th>FSRUNOUT_3</th>
      <th>ASINBILL_4</th>
      <th>MBO_MND1_1</th>
      <th>CIGSDAY_0</th>
      <th>ASINBILL_1</th>
      <th>MODERATE_ACTIVITY_8</th>
      <th>BMI_BIN_50</th>
      <th>HOURS_SLEEP_8</th>
      <th>AFLHCA17</th>
      <th>to_sort</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ONEJOB_0</th>
      <td>1.000000</td>
      <td>0.139907</td>
      <td>-0.890320</td>
      <td>-1.000000</td>
      <td>0.260725</td>
      <td>-0.140958</td>
      <td>0.000915</td>
      <td>-0.083440</td>
      <td>0.059421</td>
      <td>-0.058433</td>
      <td>-0.000145</td>
      <td>0.048282</td>
      <td>-0.039309</td>
      <td>0.011285</td>
      <td>0.057012</td>
      <td>0.085823</td>
      <td>0.085823</td>
    </tr>
    <tr>
      <th>FSNAP_1</th>
      <td>0.139907</td>
      <td>1.000000</td>
      <td>-0.122505</td>
      <td>-0.139907</td>
      <td>0.353533</td>
      <td>-0.994073</td>
      <td>0.002036</td>
      <td>-0.320934</td>
      <td>-0.161472</td>
      <td>-0.019139</td>
      <td>-0.165247</td>
      <td>0.212101</td>
      <td>-0.065813</td>
      <td>0.100248</td>
      <td>-0.019116</td>
      <td>0.114140</td>
      <td>0.114140</td>
    </tr>
    <tr>
      <th>ONEJOB_2</th>
      <td>-0.890320</td>
      <td>-0.122505</td>
      <td>1.000000</td>
      <td>0.890320</td>
      <td>-0.236158</td>
      <td>0.123406</td>
      <td>-0.013200</td>
      <td>0.077314</td>
      <td>-0.044223</td>
      <td>0.032435</td>
      <td>-0.007532</td>
      <td>-0.047123</td>
      <td>0.027308</td>
      <td>-0.009505</td>
      <td>-0.043488</td>
      <td>-0.082414</td>
      <td>0.082414</td>
    </tr>
    <tr>
      <th>WRKLYR4_0</th>
      <td>-1.000000</td>
      <td>-0.139907</td>
      <td>0.890320</td>
      <td>1.000000</td>
      <td>-0.260725</td>
      <td>0.140958</td>
      <td>-0.000915</td>
      <td>0.083440</td>
      <td>-0.059421</td>
      <td>0.058433</td>
      <td>0.000145</td>
      <td>-0.048282</td>
      <td>0.039309</td>
      <td>-0.011285</td>
      <td>-0.057012</td>
      <td>-0.085823</td>
      <td>0.085823</td>
    </tr>
    <tr>
      <th>INCGRP5_1</th>
      <td>0.260725</td>
      <td>0.353533</td>
      <td>-0.236158</td>
      <td>-0.260725</td>
      <td>1.000000</td>
      <td>-0.351563</td>
      <td>0.008154</td>
      <td>-0.277884</td>
      <td>-0.150496</td>
      <td>-0.025681</td>
      <td>-0.126563</td>
      <td>0.180663</td>
      <td>-0.059859</td>
      <td>0.045123</td>
      <td>0.004442</td>
      <td>0.089708</td>
      <td>0.089708</td>
    </tr>
    <tr>
      <th>FSNAP_2</th>
      <td>-0.140958</td>
      <td>-0.994073</td>
      <td>0.123406</td>
      <td>0.140958</td>
      <td>-0.351563</td>
      <td>1.000000</td>
      <td>-0.001383</td>
      <td>0.319773</td>
      <td>0.162123</td>
      <td>0.019483</td>
      <td>0.163559</td>
      <td>-0.211229</td>
      <td>0.067309</td>
      <td>-0.101360</td>
      <td>0.018886</td>
      <td>-0.112972</td>
      <td>0.112972</td>
    </tr>
    <tr>
      <th>MBO_SPR1_1</th>
      <td>0.000915</td>
      <td>0.002036</td>
      <td>-0.013200</td>
      <td>-0.000915</td>
      <td>0.008154</td>
      <td>-0.001383</td>
      <td>1.000000</td>
      <td>-0.022320</td>
      <td>-0.015597</td>
      <td>0.328137</td>
      <td>0.009394</td>
      <td>0.025080</td>
      <td>0.046151</td>
      <td>-0.000278</td>
      <td>-0.008108</td>
      <td>0.060662</td>
      <td>0.060662</td>
    </tr>
    <tr>
      <th>FSRUNOUT_3</th>
      <td>-0.083440</td>
      <td>-0.320934</td>
      <td>0.077314</td>
      <td>0.083440</td>
      <td>-0.277884</td>
      <td>0.319773</td>
      <td>-0.022320</td>
      <td>1.000000</td>
      <td>0.251342</td>
      <td>-0.002197</td>
      <td>0.160895</td>
      <td>-0.342797</td>
      <td>0.048829</td>
      <td>-0.087441</td>
      <td>0.039007</td>
      <td>-0.125366</td>
      <td>0.125366</td>
    </tr>
    <tr>
      <th>ASINBILL_4</th>
      <td>0.059421</td>
      <td>-0.161472</td>
      <td>-0.044223</td>
      <td>-0.059421</td>
      <td>-0.150496</td>
      <td>0.162123</td>
      <td>-0.015597</td>
      <td>0.251342</td>
      <td>1.000000</td>
      <td>-0.003460</td>
      <td>0.095375</td>
      <td>-0.293668</td>
      <td>0.040438</td>
      <td>-0.080410</td>
      <td>0.079314</td>
      <td>-0.079044</td>
      <td>0.079044</td>
    </tr>
    <tr>
      <th>MBO_MND1_1</th>
      <td>-0.058433</td>
      <td>-0.019139</td>
      <td>0.032435</td>
      <td>0.058433</td>
      <td>-0.025681</td>
      <td>0.019483</td>
      <td>0.328137</td>
      <td>-0.002197</td>
      <td>-0.003460</td>
      <td>1.000000</td>
      <td>0.006538</td>
      <td>0.004824</td>
      <td>0.056295</td>
      <td>-0.026347</td>
      <td>-0.014717</td>
      <td>0.064109</td>
      <td>0.064109</td>
    </tr>
    <tr>
      <th>CIGSDAY_0</th>
      <td>-0.000145</td>
      <td>-0.165247</td>
      <td>-0.007532</td>
      <td>0.000145</td>
      <td>-0.126563</td>
      <td>0.163559</td>
      <td>0.009394</td>
      <td>0.160895</td>
      <td>0.095375</td>
      <td>0.006538</td>
      <td>1.000000</td>
      <td>-0.124250</td>
      <td>0.030230</td>
      <td>0.011130</td>
      <td>0.034154</td>
      <td>-0.079784</td>
      <td>0.079784</td>
    </tr>
    <tr>
      <th>ASINBILL_1</th>
      <td>0.048282</td>
      <td>0.212101</td>
      <td>-0.047123</td>
      <td>-0.048282</td>
      <td>0.180663</td>
      <td>-0.211229</td>
      <td>0.025080</td>
      <td>-0.342797</td>
      <td>-0.293668</td>
      <td>0.004824</td>
      <td>-0.124250</td>
      <td>1.000000</td>
      <td>-0.040640</td>
      <td>0.058006</td>
      <td>-0.043972</td>
      <td>0.122220</td>
      <td>0.122220</td>
    </tr>
    <tr>
      <th>MODERATE_ACTIVITY_8</th>
      <td>-0.039309</td>
      <td>-0.065813</td>
      <td>0.027308</td>
      <td>0.039309</td>
      <td>-0.059859</td>
      <td>0.067309</td>
      <td>0.046151</td>
      <td>0.048829</td>
      <td>0.040438</td>
      <td>0.056295</td>
      <td>0.030230</td>
      <td>-0.040640</td>
      <td>1.000000</td>
      <td>-0.072885</td>
      <td>0.006973</td>
      <td>-0.028709</td>
      <td>0.028709</td>
    </tr>
    <tr>
      <th>BMI_BIN_50</th>
      <td>0.011285</td>
      <td>0.100248</td>
      <td>-0.009505</td>
      <td>-0.011285</td>
      <td>0.045123</td>
      <td>-0.101360</td>
      <td>-0.000278</td>
      <td>-0.087441</td>
      <td>-0.080410</td>
      <td>-0.026347</td>
      <td>0.011130</td>
      <td>0.058006</td>
      <td>-0.072885</td>
      <td>1.000000</td>
      <td>-0.028578</td>
      <td>0.049658</td>
      <td>0.049658</td>
    </tr>
    <tr>
      <th>HOURS_SLEEP_8</th>
      <td>0.057012</td>
      <td>-0.019116</td>
      <td>-0.043488</td>
      <td>-0.057012</td>
      <td>0.004442</td>
      <td>0.018886</td>
      <td>-0.008108</td>
      <td>0.039007</td>
      <td>0.079314</td>
      <td>-0.014717</td>
      <td>0.034154</td>
      <td>-0.043972</td>
      <td>0.006973</td>
      <td>-0.028578</td>
      <td>1.000000</td>
      <td>-0.038439</td>
      <td>0.038439</td>
    </tr>
    <tr>
      <th>AFLHCA17</th>
      <td>0.085823</td>
      <td>0.114140</td>
      <td>-0.082414</td>
      <td>-0.085823</td>
      <td>0.089708</td>
      <td>-0.112972</td>
      <td>0.060662</td>
      <td>-0.125366</td>
      <td>-0.079044</td>
      <td>0.064109</td>
      <td>-0.079784</td>
      <td>0.122220</td>
      <td>-0.028709</td>
      <td>0.049658</td>
      <td>-0.038439</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



The below cell translates the feature names into sensible descriptions for plotting.


```python
# Creating a dictionary for graphing the most important features in a bar chart.
labels = {'ASINBILL_1': 'Bills: Very Worried', 
          'WRKLYR4_2': 'No Job Past 12 Months',
          'SMKNOW_1': 'Every Day Smoker',
          'ONEJOB_2': 'Did Not Have 2+ Jobs',
          'ASINBILL_4': 'Bills: Not Worried at All',
          'SMKNOW_0': 'Non-Smoker',
          'R_MARITL_1': 'Married: Spouse in Househole',
          'PDSICKA_1': 'Paid Sick Leave',
          'PDSICKA_2': 'No Paid Sick Leave',
          'ASINKNT_4': 'Tight-Knit Neighborhood: Definitely Disagree',
          'MBOMND_2': 'No Mindfulness Meditation',
          'YRSWRKPA_1':'Working for 1 Year',
          'MBO_IMG_1': 'Use Guided Imagery Meditation',
          'YRSWRKPA_50': 'Working for 35+ Years',
          'FSRUNOUT_2': 'Worried Food Will Run Out: Often',
          'FSRUNOUT_1': 'Worried Food Will Run Out: Sometimes',
          'FM_ELDR_0': 'No 65+ Females in Home',
          'ONEJOB_0': 'No Job',
          'VIGOROUS_ACTIVITY_100': 'Vigorous Activity: Never',
          'INCGRP5_4': 'Income: $100,000+',
          'INCGRP5_1': 'Income: Below $35k',
          'INCGRP5_2': 'Income: $35k - $75k',
          'MBO_SPR1_1': 'Use Spiritual Meditatation',
          'MBO_SPR1_2': 'No Spiritual Meditation',
          'FSRUNOUT_3': 'Worried Food Will Run Out: Never',
          'WRKLYR4_0': 'Has Job',
          'FSNAP_1': 'Receives Food Stamps',
          'FSNAP_2': 'No Food Stamps',
          'MBO_MND1_1': 'Uses Mindfulness Meditation',
          'MBO_MND1_2': 'Doesn\t Use Mindfulness Meditation',
          'BMI_BIN_50': 'BMI above 33 (top 20%)',
          'CIGSDAY_0': 'Non-Smoker',
          'MODERATE_ACTIVITY_8': 'Moderate Phys Activity 4-8 times/week',
          'VIGOROUS_ACTIVITY_8': 'Vigorous Phys Activity 4-8 times/week',
          'ALCOHOL_PER_DAY_DRANK_1': '1 Drinks on Days that Drank',
          'ALCOHOL_PER_DAY_DRANK_0': '0 Drinks on Days that Drank',
          'YEARS_WORKED_1': '1 Year at Present Job',
          'R_MARITL_4': 'Marital Status: Widowed',
          'R_MARITL_7': 'Marital Status: Never Married',
          'ASISIM_0': 'Male: Straight, that is, Not Gay',
          'MBO_IMG1_2': 'Doesn\t Use Guided Imagery Meditation',
          'WRKCATA_1': 'Private Company Employee',
          'MRACRPI2_1': 'Race: White',
          'ASISIF_0': 'Female: Straight, that is, Not Gay',
          'ASISIM_2': 'Male: Straight, that is, Not Gay',
          'HOURS_SLEEP_8': '8 Hours Sleep',
          'ALC12MWK_95': 'Did Not Drink Past Year',
          'RAT_CAT5_14': 'Income Ratio to Poverty Threshold: 5.00+',
          'ASINKNT_2': 'Tight-Knit Neighborhood: Somewhat Agree',
          'VIMGLASS_1': 'Wears Glasses or Contact Lenses',
          'MRACRPI2_2': 'Race: Black',
          'VIMGLASS_2': 'Does Not Wear Glasses or Contact Lense',
          'RAT_CAT5_2': 'Income Ratio to Poverty Threshold: .5 - .74',
          'RAT_CAT5_8': 'Income Ratio to Poverty Threshold: 2.0 - 2.49',
          'R_MARITL_5': 'Marital Status: Divorced',
          'AGE_83': 'Age 73 to 83',
          'SUPERVIS_2': 'Does Not Supervise Employees',
          'SUPERVIS_1': 'Does Supervise Employees',
          'MODERATE_ACTIVITY_100': 'Moderate Activity: 30+ Hours/week',
          'BMI_BIN_4': 'BMI 29 - 33',
          'REGION_4': 'Region: West',
          'REGION_3': 'Region: South',
          'HOURS_SLEEP_5': '5 Hours Sleep or Less',
          'HOURS_SLEEP_7': '7 Hours Sleep',
          'ASINBILL_2': 'Bills: Moderatly Worried',
          'FM_KIDS_0': 'No Children Under 18',
          'AWEBEML_1': 'Uses e-mail',
          'AWEBEML_2': 'Does Not Use e-mail',
          'FM_SIZE_1': 'Family Size: 1',
          'FINTR1YN_2': 'No Interest Bearing Accounts, Funds etc.',
          'WEB_HOURS_1': 'Light Internet Use',
          'FM_ELDR_2': '2 Females over 65 in Household',
          'WEB_HOURS_5': 'Medium Internet Use',
          'PAR_STAT_3': 'No Children in Household'
         }
```


```python
# Creating list of labels from dictionary to apply to graph.
to_sort_df = to_sort_df.iloc[::-1]
plot_labels = [labels[x] for x,y in to_sort_df.iterrows()]
```

The below graph will demonstrate positive or negative corralation between target and feature.


```python
# I'd like to have different colors for pos vs neg.
to_sort_df['positive'] = to_sort_df['AFLHCA17'] > 0
plt.figure(figsize=(7,10))

fig.patch.set_facecolor((1,1,1,1))
font = {'color': 'white'}

# plot_labels was established early in the exploration when using most important 
# features to guide modeling.
plt.barh(plot_labels, to_sort_df['AFLHCA17'], 
         color=to_sort_df.positive.map({True: 'green', False: 'orange'}), 
         edgecolor='white', lw=1);



plt.title('Correlation of 15 Most Important Features')
plt.ylabel('Survey Question Description')
plt.xlabel('Correlation of variable to Depression/Anxiety')
plt.show();



```


![png](output_62_0.png)


## Individual Testing

Kept for posterity. This was intended to test to see if some of the categorical variables would be better suited non-categorical use. It didn't work out, so I proceeded with one-hotted versions.


```python
tiny_test = combined_sig_data[['YEARS_WORKED', 'VIGOROUS_ACTIVITY', 'MODERATE_ACTIVITY',
                          'ALCOHOL_PER_DAY_DRANK', 'BMI_BIN', 'HOURS_SLEEP', 'WEB_HOURS']]
```


```python
tiny_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEARS_WORKED</th>
      <th>VIGOROUS_ACTIVITY</th>
      <th>MODERATE_ACTIVITY</th>
      <th>ALCOHOL_PER_DAY_DRANK</th>
      <th>BMI_BIN</th>
      <th>HOURS_SLEEP</th>
      <th>WEB_HOURS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>8</td>
      <td>100</td>
      <td>3</td>
      <td>33</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>100</td>
      <td>3</td>
      <td>26</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>50</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>50</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>100</td>
      <td>3</td>
      <td>3</td>
      <td>33</td>
      <td>8</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The existing values must first be converted into integers in order to be encoded.
tiny_test = tiny_test.astype(int)
tiny_dummies = pd.get_dummies(tiny_test, 
                         columns=list(tiny_test.columns),
                         drop_first=False)
```


```python
tiny_dummies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEARS_WORKED_1</th>
      <th>YEARS_WORKED_2</th>
      <th>YEARS_WORKED_5</th>
      <th>YEARS_WORKED_10</th>
      <th>YEARS_WORKED_15</th>
      <th>YEARS_WORKED_20</th>
      <th>YEARS_WORKED_30</th>
      <th>YEARS_WORKED_50</th>
      <th>YEARS_WORKED_1000</th>
      <th>VIGOROUS_ACTIVITY_0</th>
      <th>VIGOROUS_ACTIVITY_1</th>
      <th>VIGOROUS_ACTIVITY_3</th>
      <th>VIGOROUS_ACTIVITY_8</th>
      <th>VIGOROUS_ACTIVITY_15</th>
      <th>VIGOROUS_ACTIVITY_30</th>
      <th>VIGOROUS_ACTIVITY_100</th>
      <th>MODERATE_ACTIVITY_0</th>
      <th>MODERATE_ACTIVITY_1</th>
      <th>MODERATE_ACTIVITY_3</th>
      <th>MODERATE_ACTIVITY_8</th>
      <th>MODERATE_ACTIVITY_15</th>
      <th>MODERATE_ACTIVITY_30</th>
      <th>MODERATE_ACTIVITY_100</th>
      <th>ALCOHOL_PER_DAY_DRANK_0</th>
      <th>ALCOHOL_PER_DAY_DRANK_1</th>
      <th>ALCOHOL_PER_DAY_DRANK_3</th>
      <th>ALCOHOL_PER_DAY_DRANK_5</th>
      <th>ALCOHOL_PER_DAY_DRANK_8</th>
      <th>ALCOHOL_PER_DAY_DRANK_12</th>
      <th>ALCOHOL_PER_DAY_DRANK_20</th>
      <th>ALCOHOL_PER_DAY_DRANK_100</th>
      <th>BMI_BIN_23</th>
      <th>BMI_BIN_26</th>
      <th>BMI_BIN_29</th>
      <th>BMI_BIN_33</th>
      <th>BMI_BIN_50</th>
      <th>HOURS_SLEEP_5</th>
      <th>HOURS_SLEEP_6</th>
      <th>HOURS_SLEEP_7</th>
      <th>HOURS_SLEEP_8</th>
      <th>HOURS_SLEEP_9</th>
      <th>HOURS_SLEEP_12</th>
      <th>HOURS_SLEEP_15</th>
      <th>HOURS_SLEEP_100</th>
      <th>WEB_HOURS_0</th>
      <th>WEB_HOURS_1</th>
      <th>WEB_HOURS_5</th>
      <th>WEB_HOURS_10</th>
      <th>WEB_HOURS_20</th>
      <th>WEB_HOURS_40</th>
      <th>WEB_HOURS_1000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
grid = {'criterion': ['gini', 'entropy'],
        'max_depth': [3,6,7,8],
        'max_features': [30,50,200],
        'gamma': [5,10,15,20],
        'max_features': [30,50,300]} 
```


```python
best_params_xg = grid_search(XGBClassifier, tiny_test[:26311], test_target, grid,
                          verbose=False, n_estimators=50, scoring='accuracy')
```

    Mean Training Score: 52.37%
    Mean Test Score: 68.87%
    Best Parameter Combination Found During Grid Search:
    {'criterion': 'gini', 'gamma': 5, 'max_depth': 3, 'max_features': 30}



```python
test = tree_builder(tiny_test[:26311], test_target, params=best_params_xg,
             classifier=XGBClassifier)
```

    Train Accuracy:  0.7009577864490955
    Test Accuracy:  0.6886591669200365
    Precision:  0.024279210925644917
    Recall:  0.2874251497005988
    f1_score:  0.04477611940298507
    
    AUC is :0.49



![png](output_71_1.png)


    HOURS_SLEEP              0.444926
    YEARS_WORKED             0.341876
    ALCOHOL_PER_DAY_DRANK    0.213199
    WEB_HOURS                0.000000
    BMI_BIN                  0.000000
    MODERATE_ACTIVITY        0.000000
    VIGOROUS_ACTIVITY        0.000000
    dtype: float32



![png](output_71_3.png)


# Modeling

## Main Target: Depression/Anxiety Duration

### Decision Tree Classifier


```python
grid = {'criterion': ['gini', 'entropy'],
        'max_depth': [2,3,6,7,8],
        'min_samples_split': [2,5,10,15,20,30],
        'min_samples_leaf': [2,5,10,15,20,30],
        'max_features': [20,30,50,100]} 
```


```python
best_params_dtc = grid_search(DecisionTreeClassifier, trim_dummies, test_target, 
                          grid, verbose=False)
```

    Mean Training Score: 72.74%
    Mean Test Score: 68.00%
    Best Parameter Combination Found During Grid Search:
    {'criterion': 'entropy', 'max_depth': 8, 'max_features': 30, 'min_samples_leaf': 15, 'min_samples_split': 10}



```python
tree_builder(trim_dummies, test_target, params=best_params_dtc)
```

    Train Accuracy:  0.6626970050169766
    Test Accuracy:  0.6606871389480086
    Precision:  0.0508916920400174
    Recall:  0.7005988023952096
    f1_score:  0.09489051094890512
    
    AUC is :0.68



![png](output_77_1.png)


    FSNAP_2                  0.202019
    WRKLYR4_2                0.083993
    MBO_SPR1_1               0.079041
    ONEJOB_0                 0.062988
    MBO_MND1_1               0.046803
    FSRUNOUT_3               0.045223
    MRACRPI2_1               0.037444
    MODERATE_ACTIVITY_100    0.035658
    ASISIF_0                 0.029627
    FM_SIZE_1                0.026986
    dtype: float64



![png](output_77_3.png)





![png](output_77_4.png)



### Random Forest Model with Grid Search


```python
grid = {'criterion': ['gini', 'entropy'],
        'max_depth': [2,3,6,7,8],
        'min_samples_split': [2,5,10,15,20,30],
        'min_samples_leaf': [2,5,10,15,20,30],
        'max_features': [20,30,50,100]} 
```


```python
best_params_rf = grid_search(RandomForestClassifier, trim_dummies, test_target, 
                          grid, verbose=False, n_estimators=100)
```

    Mean Training Score: 76.62%
    Mean Test Score: 71.04%
    Best Parameter Combination Found During Grid Search:
    {'criterion': 'entropy', 'max_depth': 8, 'max_features': 20, 'min_samples_leaf': 2, 'min_samples_split': 10}



```python
best_features_rfc = tree_builder(trim_dummies, test_target, 
                                 params=best_params_rf,
                                 classifier=RandomForestClassifier)
```

    Train Accuracy:  0.7214311052551563
    Test Accuracy:  0.7146549103070842
    Precision:  0.06063720452209661
    Recall:  0.7065868263473054
    f1_score:  0.11168954093705633
    
    AUC is :0.71



![png](output_81_1.png)


    ONEJOB_0      0.048666
    FSNAP_1       0.034362
    ONEJOB_2      0.034311
    WRKLYR4_0     0.028879
    INCGRP5_1     0.028834
    FSNAP_2       0.028325
    MBO_SPR1_1    0.026456
    FSRUNOUT_3    0.025344
    ASINBILL_4    0.024452
    MBO_MND1_1    0.023479
    dtype: float64



![png](output_81_3.png)


### XGBoost Model with Grid Search


```python
# Create a spectrum of values to test against.
grid = {'criterion': ['gini', 'entropy'],
        'max_depth': [3,6,7,8],
        'max_features': [30,50,200],
        'gamma': [5,10,15,20],
        'max_features': [30,50,300]} 

# grid_search function returns the params to be used in the model.
best_params_xg = grid_search(XGBClassifier, trim_dummies, test_target, grid,
                          verbose=False, n_estimators=50, scoring='accuracy')
```

    Mean Training Score: 76.23%
    Mean Test Score: 71.71%
    Best Parameter Combination Found During Grid Search:
    {'criterion': 'gini', 'gamma': 5, 'max_depth': 3, 'max_features': 30}



```python
# This is the function that creates the modeling and plotting.
best_features = tree_builder(trim_dummies, test_target, params=best_params_xg,
             classifier=XGBClassifier)
```

    Train Accuracy:  0.716566158212132
    Test Accuracy:  0.7170872605655214
    Precision:  0.060228452751817235
    Recall:  0.6946107784431138
    f1_score:  0.11084567606306736
    
    AUC is :0.71



![png](output_84_1.png)


    WRKLYR4_0     0.076152
    FSRUNOUT_3    0.069124
    INCGRP5_1     0.052336
    FM_ELDR_2     0.043366
    FSNAP_2       0.034605
    ASINBILL_4    0.031904
    CIGSDAY_0     0.027383
    MBO_MND1_1    0.027180
    FM_ELDR_0     0.025769
    ASINBILL_1    0.024286
    dtype: float32



![png](output_84_3.png)


### Support Vector Classifier

**Best Model**

This was the one that performed the best. With true negative and true values both almost 3/4, it showed a healthy degree of utility. With a target size of 750 compared to a total sample size of 26,000, it was a bit like finding a needle in a haystack, but the results have worked out modestly well.


```python
grid = {'kernel': ['linear','poly','sigmoid', 'rbf'],
        'C': [4,5,6,7,8],
        'gamma': ['auto', 'scale']} 
```

Grid search for parameters. Commenting out so it won't take up computing time when re-running notebook.


```python
best_params_svc = grid_search(SVC, trim_dummies, test_target,
                          grid, verbose=False, scoring='accuracy')
```

    Mean Training Score: 82.08%
    Mean Test Score: 74.63%
    Best Parameter Combination Found During Grid Search:
    {'C': 6, 'gamma': 'auto', 'kernel': 'sigmoid'}



```python
# Backup params in case I don't want to re-run grid search.
# svc_params = {'C': 6, 'gamma': 'auto', 'kernel': 'sigmoid'}
```


```python
tree_builder(trim_dummies, test_target, params=best_params_svc,
             classifier=SVC)
```

    Train Accuracy:  0.743323366948766
    Test Accuracy:  0.746275463666768
    Precision:  0.06938073394495413
    Recall:  0.7245508982035929
    f1_score:  0.12663526949241236
    
    AUC is :0.79



![png](output_91_1.png)



![png](output_91_2.png)


### K Nearest Neighbors


```python
# Backup params in case I don't want to re-run grid search.
# knn_params = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
```


```python
best_params_knn = grid_search(KNeighborsClassifier, trim_dummies, test_target,
                          knn_params, verbose=False, scoring='recall')
```

    Mean Training Score: 67.32%
    Mean Test Score: 62.87%
    Best Parameter Combination Found During Grid Search:
    {'n_neighbors': 17}



```python
tree_builder(trim_dummies, test_target, params=best_params_knn,
             classifier=KNeighborsClassifier)
```

    Train Accuracy:  0.7788476156691836
    Test Accuracy:  0.766798418972332
    Precision:  0.06658211794546608
    Recall:  0.6287425149700598
    f1_score:  0.12041284403669725
    
    AUC is :0.7



![png](output_95_1.png)


## Side Targets

I did a few additional targets on other subjects to find out if there was anything worthwhile.

There wasn't really much here, but I'm keeping it in as a means of demonstrating repeatability. The functions I've created make displaying a fairly complete analysis of each target very simple to do. 

### Bipolar, Schizophrenia etc.

The sample size for the target values was far too small to be useful.


```python
target_bi = cdc_ad['AFLHCA17']
target_bi = target_bi.fillna(0)
target_bi = target_bi.where(target_bi == 1, 0)
```


```python
target_bi = target_bi.fillna(0)
target_bi = target_bi.where(target_bi == 1, 0)
```


```python
target_bi = target_bi[:26311]
```

Grid search for parameters. Commenting out so it won't take up computing time when re-running notebook.


```python
grid = {'kernel': ['linear','poly','sigmoid', 'rbf'],
        'C': [.001,5,6,7,8],
        'gamma': ['auto', 'scale']} 

best_params_svc = grid_search(SVC, trim_dummies, target_bi,
                          grid, verbose=False, scoring='f1')
```

    Mean Training Score: 65.61%
    Mean Test Score: 6.09%
    Best Parameter Combination Found During Grid Search:
    {'C': 7, 'gamma': 'auto', 'kernel': 'poly'}



```python
bi_params = {'kernel': 'poly',
            'C': .001,
            'gamma': 'scale'} 
```


```python
tree_builder(trim_dummies, target_bi, params=best_params_svc,
             classifier=SVC)
```

    Train Accuracy:  0.4033851923174378
    Test Accuracy:  0.3957129826695044
    Precision:  0.031994047619047616
    Recall:  0.6417910447761194
    f1_score:  0.06094968107725018
    
    AUC is :0.52



![png](output_106_1.png)



![png](output_106_2.png)


### Kidney Problems

Another small sample size, such that no conclusions could be found.


```python
target_kid = cdc_ad['KIDWKYR']

target_kid = target_kid.fillna(0)
target_kid = target_kid.where(target_kid == 1, 0)

target_kid = target_kid[:26311]
```

Grid search for parameters. Commenting out so it won't take up computing time when re-running notebook.


```python
grid = {'kernel': ['linear','poly','sigmoid', 'rbf'],
        'C': [4,5,6,7,8],
        'gamma': ['auto', 'scale']} 

best_params_svc = grid_search(SVC, trim_dummies, target_kid,
                          grid, verbose=False, scoring='accuracy')
```

    Mean Training Score: 72.00%
    Mean Test Score: 50.61%
    Best Parameter Combination Found During Grid Search:
    {'C': 7, 'gamma': 'auto', 'kernel': 'linear'}



```python
tree_builder(trim_dummies, target_kid, params=best_params_svc,
             classifier=SVC)
```

    Train Accuracy:  0.5131505599756753
    Test Accuracy:  0.506080875646093
    Precision:  0.02293151533932445
    Recall:  0.43529411764705883
    f1_score:  0.04356785398881366
    
    AUC is :0.45



![png](output_112_1.png)



![png](output_112_2.png)


### Heavy Drinking

Not sure why this one didn't work out, aside from the fact that the data set was culled with mental illness in mind rather than drinking, but many of the features should have been the same. Something I would have explored with more time, but it wasn't the primary interest.


```python
# There had to be some re-arranging for this one since there were several 
# related columns in the original data set.
alc_combined = cdc_combined.drop(['ALC12MWK', 'ALCOHOL_PER_DAY_DRANK'], 
                                axis = 1)

alc_combined = alc_combined.astype(int)
drunk_dummies = pd.get_dummies(alc_combined, 
                         columns=list(alc_combined.columns),
                         drop_first=False)

trim_dummies = drunk_dummies[drunk_dummies.columns[:40]]

# average drinks per day, days per year, days per week.
cdc_alc = cdc_ad[['ALCAMT', 'ALC12MYR', 'ALC12MWK']].copy()
```

Find out how many heavy drinkers there are. Will need to do some feature engineering to split this up by sex.


```python
cdc_alc[(cdc_alc['ALCAMT'] * cdc_alc['ALC12MYR']) >= 430].count() # 780 for men
```




    ALCAMT      2354
    ALC12MYR    2354
    ALC12MWK    2354
    dtype: int64




```python
cdc_alc['DRINKS_PER_YEAR'] = cdc_alc['ALCAMT'] * cdc_alc['ALC12MYR']

cdc_alc['DRINKS_PER_YEAR'].fillna(0, inplace=True)

target_alc = cdc_alc.DRINKS_PER_YEAR.map(lambda x: 1 if x > 430 else 0)
# [for cdc_alc.map('ALCAMT' * 'ALC12MYR' >= 780, 0)

target_alc = target_alc[:26311]
```

Grid search for parameters. Commenting out so it won't take up computing time when re-running notebook.


```python
grid = {'kernel': ['linear','poly','sigmoid', 'rbf'],
        'C': [4,5,6,7,8],
        'gamma': ['auto', 'scale']} 

best_params_svc = grid_search(SVC, trim_dummies, target_alc,
                          grid, verbose=False, scoring='recall')
```

    Mean Training Score: 55.05%
    Mean Test Score: 56.86%
    Best Parameter Combination Found During Grid Search:
    {'C': 4, 'gamma': 'auto', 'kernel': 'poly'}



```python
svc_params = {'C': 6, 'gamma': 'auto', 'kernel': 'sigmoid'}
```


```python
tree_builder(trim_dummies, target_alc, params=best_params_svc,
             classifier=SVC)
```

    Train Accuracy:  0.4245679825672731
    Test Accuracy:  0.42566129522651264
    Precision:  0.0890960890960891
    Recall:  0.5685950413223141
    f1_score:  0.1540528437080161
    
    AUC is :0.5



![png](output_122_1.png)



![png](output_122_2.png)


# Conclusions

## Summary

There were five models generated for the target data of individuals classified as having dibilitating depression or anxiety. The decision tree was the first, as it was an easy model from which to pull robust metrics and visuals. The results were fine but not fantastic. The next model was the random forest, which definitely improved the output figures. A natural progression led to the XGBoost model, which made further improvments still. It was followed up with the Support Vector Classifier, which was the model I identified for the best model for my purposes. After that came the K Nearest Neighbors model, which actually had a higher accuracy, but the the AUC was higher on the SVC, and recall was superior, which I felt was important in maximizing positive hits as opposed with avoiding false positives. Better to capture as many at-risk individuals as possible.

As for the final output, I had an accuracy value of .74 with the Support Vector Classifier. This model was able to create a definite prediction regarding debilitating depression and anxiety. True positive and true negative values relative to their false counterparts were both nearly 75%. The f1 score was also superior to the other models at 12.6, which sounds terrible but it was a very small target in comparison with the overall sample. While there were a sizeable number of false positives, the model caught the majority of mentally ill individuals.

With some strong tools and enough data, mental illness, and depression and anxiety in particular, can be predicted to a fairly strong extent based on economic factors and personal lifestyles. While it can't necessarily be ascertained which causes which -- whether, for example, individuals have difficulty maintining employment because they have debilitating depression and/or anxity or if they have debilitating depression and/or anxity because they are unemployed -- that distinction does not necessarily matter. While more research would need to be done to investigate and dissect unique factors (such as unemployment or poverty), a report such as this can be used as a tool to assist in determining which features to pursue, as well as carving out segments of the popoulation that are more likely to be at risk and allocating budgets for mental health intervention appropriately.

Further, because so many of the most important factors were economic in nature, it may be indicative of the need to address poverty issues one way or another. Whether poverty causes debilitating depression and anxiety or depression and anxiety cause poverty is beside the point; the reality is that these forms of mental illness exist under those circumstances, for one reason or another, and should be addressed accordingly.

Though I am includeing below the top 50 most important features used in identifying debilitating depression and anxiety in this report, the individual variables should not be used on their own to make judgments about any particular aspect, especially if the correlation is close to center. This report is the culmination of 100 separate variables, and there is planty of margin for error in the details. For example, black race shows just left of center as being at less risk for debilitating depression and anxiety, while white race shows just to the right as being slightly more at risk for debilitating depression and anxiety. This is not an argument that racial injustice has been alleviated and people with African heritage have finally reached parity with those of Eurpoean heritage. The only question this report is equipped to answer is who is more at risk for debilitating depression and anxiety; any other arguments about relative happiness or well being are far beyong the scope of the data herein.

## Important Feature Plot


```python
# Put the target together with the original dummy set to establish correlation
# for plotting.
corr_dummies = pd.concat([you_dummies[best_features_rfc.index[:50]], 
                          test_target], axis = 1)
corr_dummies = corr_dummies.corr()
corr_dummies['to_sort'] = abs(corr_dummies['AFLHCA17'])
# Take everything but the target.
to_sort_df = corr_dummies[['AFLHCA17', 'to_sort']][:-1]
# Reverse the order for sorting
to_sort_df = to_sort_df.iloc[::-1]
plot_labels = [labels[x] for x,y in to_sort_df.iterrows()]
```


```python
# I'd like to have different colors for pos vs neg.
to_sort_df['positive'] = to_sort_df['AFLHCA17'] > 0
fig = plt.figure(figsize=(10, 17))
ax = fig.add_subplot(111)
fig.patch.set_facecolor((0,0,0,0))

# plot_labels was established early in the exploration when using most 
# important features to guide modeling.
plt.barh(plot_labels, to_sort_df['AFLHCA17'], 
         color=to_sort_df.positive.map({True: 'green', False: 'orange'}), 
                                        edgecolor='black', lw=1)
plt.title('Correlation of 50 Most Important Features', color='black', fontsize=30)
# ax.xaxis.label.set_color('white')
# ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='black', labelsize=16)
ax.tick_params(axis='y', colors='black', labelsize=16)
plt.xlabel('Correlation of Feature to Depression/Anxiety', color='black', 
           fontsize=22)
plt.ylabel('Survey Question Descripition', color='black', fontsize=25)
plt.show()
```


![png](output_128_0.png)



```python

```
