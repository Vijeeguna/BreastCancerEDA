import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import time

# Task 1 - Importing Data
from scipy.stats import spearmanr, stats

data = pd.read_csv('data.csv')
print(pd.DataFrame.head(data, 5))

# Task 2 - Separate Target from Features
print(data.info())
# Helpful to determine if we need to standardize or normalize feature before feature selection?
print(data.describe())
# To view all column names
print(data.columns)  #
# Saving targer in Y
y = data['diagnosis']
# Columns to be dropped
drop_col = ['Unnamed: 32', 'id', 'diagnosis']
x = data.drop(drop_col, axis=1)
print(x.columns)

# Task 3: Plotting distribution of the target variable - Diagnosis
# Class imbalance problem?

ax = sns.countplot(y, label='Counts')
B, M = y.value_counts()
print('The number of Benign tumors is: ', B)
print('The number of Malignant tumors is: ', M)
# Uncomment below to view the graph
# plt.show()
# This shows a class problem. More benign tumor than malignant tumor data
# It's always simpler to work on a classification problem when the different classes are equally represented.
# Here, the count ratio between Malignant tumorts and Benign tumors is close to 0.6, which is acceptable.
# This is important to determine the sampling type
# Should we oversample malignant or undersample benign?

# Task 4: Visulaizing data with seaborn
# Violin plots are like box plots but they show the probability density  of the data at different values
# smoothed over by a kernel density estimator

# To check if there are any missing values
x.isnull().values.any()
# Should return false

# Standardize data
x_std = (x.mean() - x) / x.std()
# Normalizing data so all feature values are in scale
# Otherwise machine learning algorithms do not converge well
# Normalisation needs to be done before visualization, feature selection, classification, etc.

df1 = pd.concat([y, x_std.iloc[:, 0:10]], axis=1)
# For conversion from long format to wide format
df1 = pd.melt(df1, id_vars='diagnosis',
              var_name='features',
              value_name='value')
# Plotting logic
plt.figure(figsize=(10, 10))
sns.violinplot(x='features', y='value', hue='diagnosis', data=df1, split=True, inner='quart')
plt.xticks(rotation=45)

df2 = pd.concat([y, x_std.iloc[:, 10:20]], axis=1)
# For conversion from long format to wide format
df2 = pd.melt(df2, id_vars='diagnosis',
              var_name='features',
              value_name='value')
# Plotting logic
plt.figure(figsize=(10, 10))
sns.violinplot(x='features', y='value', hue='diagnosis', data=df2, split=True, inner='quart')
plt.xticks(rotation=45)

df3 = pd.concat([y, x_std.iloc[:, 20:30]], axis=1)
# For conversion from long format to wide format
df3 = pd.melt(df3, id_vars='diagnosis',
              var_name='features',
              value_name='value')
# Plotting logic
plt.figure(figsize=(10, 10))
sns.violinplot(x='features', y='value', hue='diagnosis', data=df3, split=True, inner='quart')
plt.xticks(rotation=45)

# Tip: When considering feature selection
# If the median (dotted line) are distinct, the feature is useful for classification
# If the median line is not distinct, they may not be help for classification

# Plots that are similar might be correlated

# To check for Outliers, we use a boxplot
plt.figure(figsize=(10, 10))
sns.boxplot(x='features', y='value', hue='diagnosis', data=df1)
plt.xticks(rotation=45)

plt.figure(figsize=(10, 10))
sns.boxplot(x='features', y='value', hue='diagnosis', data=df2)
plt.xticks(rotation=45)

plt.figure(figsize=(10, 10))
sns.boxplot(x='features', y='value', hue='diagnosis', data=df3)
plt.xticks(rotation=45)

# Task 6: Using joint plots for feature comparison
j = sns.jointplot(x.loc[:, 'concavity_worst'],
                  x.loc[:, 'concave points_worst'],
                  kind='regg',
                  color='red')
j.annotate(stats.pearsonr)
# To add the pearson number in the plot

# Task 7
# Swarm plots

sns.set(style='whitegrid', palette='muted')
plt.figure(figsize=(10, 10))
sns.swarmplot(x='features', y='value', hue='diagnosis', data=df1)
plt.xticks(rotation=45)

plt.figure(figsize=(10, 10))
sns.swarmplot(x='features', y='value', hue='diagnosis', data=df2)
plt.xticks(rotation=45)

plt.figure(figsize=(10, 10))
sns.swarmplot(x='features', y='value', hue='diagnosis', data=df3)
plt.xticks(rotation=45)

# From looking at the swarm graph we can tell that perimeter worst is a better predictor that say
# smoothness worst

# Task 8
# Looking at pair-wise correlation across all feature variables
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=0.5, fmt='.1f', ax=ax)
# plt.show()

# Detemining performance criterion
# Choosing Recall as opposed to Accuracy
# Number of breast cancer cases identified correctly by total number of existing cases in the sample
# We pick recall since it far more important to not get a FALSE NEGATIVE

# FEATURE SELECTION
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score
from sklearn.feature_selection import RFE

# Manual Feature Selection based on EDA
x_man = data[['radius_mean', 'texture_mean', 'smoothness_mean', 'concave points_mean',
          'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
          'smoothness_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
          'radius_worst', 'texture_worst', 'smoothness_worst', 'concave points_worst',
          'symmetry_worst', 'fractal_dimension_worst']]

# recall_score needs y in binary format
y = y.replace('B', 0)
y = y.replace('M', 1)
# Random state ensures the split is reproducible much that set.seed()
x_train, x_test, y_train, y_test = train_test_split(x_man, y, test_size=0.3, random_state=42)

man_rf = RandomForestClassifier(random_state=43)
clf_man = man_rf.fit(x_train, y_train)
man_pred = clf_man.predict(x_test)
recall_man = recall_score(y_test, man_pred)
print('Recall is: ', recall_man)
acc_man = accuracy_score(y_test, man_pred)
print('Accuracy is: ', acc_man)
f1_man = f1_score(y_test, man_pred)
print('F1 score is: ', f1_man)
cm_man = confusion_matrix(y_test, man_pred)
print(cm_man)

# Recursive Feature Elimination with Random Forest
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)
clf_rfe = RandomForestClassifier(random_state=43)
rfe = RFE(estimator=clf_rfe, n_features_to_select=18, step=1)
rfe = rfe.fit(x_train, y_train)
print('Chosen best 18 feature by rfe:',x_train.columns[rfe.support_])

# Interestingly 3 highly correlated variables are chosen together
# The features we had eliminated in the manual feature selection
# Suspicious

rfe_pred = rfe.predict(x_test)
recall_rfe = recall_score(y_test, rfe_pred)
print('Recall is: ', recall_rfe)
acc_rfe = accuracy_score(y_test, rfe_pred)
print('Accuracy is: ', acc_rfe)
f1_rfe = f1_score(y_test, rfe_pred)
print('F1 score is: ', f1_rfe)
cm_rfe = confusion_matrix(y_test, rfe_pred)
print(cm_rfe)

#Accuracy is better. But recall is the same

# Optimal number of feature selection
from sklearn.feature_selection import RFECV

clf_rfe_cv = RandomForestClassifier(random_state=43)

# Scoring tells the algorithm what metrics to use to elimiate features
rfecv = RFECV(estimator= clf_rfe_cv, step = 1, cv = 10, scoring='recall')
rfecv = rfecv.fit(x_train, y_train)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])

x_opt = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'concavity_mean', 'concave points_mean', 'area_se', 'radius_worst',
       'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst']]
x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size= 0.3, random_state=42)
opt_mod = RandomForestClassifier(random_state=43)
opt = opt_mod.fit(x_train, y_train)

opt_pred = opt.predict(x_test)
recall_opt = recall_score(y_test, opt_pred)
print('Recall for optimal model is: ', recall_opt)
acc_opt = accuracy_score(y_test, opt_pred)
print('Accuracy for optimal model  is: ', acc_opt)
f1_opt = f1_score(y_test, opt_pred)
print('F1 score for optimal model is: ', f1_opt)
cm_opt = confusion_matrix(y_test, opt_pred)
print(cm_opt)

import matplotlib.pyplot as plt
plt.figure()
plt.xlabel('Number of Features')
plt.ylabel('Cross validation score')
# This score evaluates the performance of your model when using  n number of features selection
plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
# rfecv.grid_scores_ returns cross validation score
#plt.show()


# KNN for classification

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

knn = KNeighborsClassifier(n_neighbors= 11)
knn_mod = knn.fit(x_train, y_train)
knn_predict = knn_mod .predict(x_test)
knn_cm = confusion_matrix(y_test, knn_predict)
acc_knn = accuracy_score(y_test, knn_predict)
print('Accuracy for KNN model with 11 N is: ', acc_knn)
recall_knn = recall_score(y_test, knn_predict)
print('Recall for KNN model with 11 N is is: ', recall_knn)
print(knn_cm)

recall_val = []
acc_val = []
for k in range(1, 30):
    print('For ', k, 'nearest neighbours')
    knn = KNeighborsClassifier(n_neighbors=k)
    knn_mod = knn.fit(x_train, y_train)
    knn_predict = knn_mod.predict(x_test)
    knn_cm = confusion_matrix(y_test, knn_predict)
    acc_knn = accuracy_score(y_test, knn_predict)
    print('Accuracy for KNN model  is: ', acc_knn)
    recall_knn = recall_score(y_test, knn_predict)
    print('Recall for KNN model  is: ', recall_knn)
    print(knn_cm)
    recall_val.append(recall_knn)
    acc_val.append(acc_knn)

acc_curve=pd.DataFrame(acc_val)
acc_curve.plot()
recall_curve=pd.DataFrame(recall_val)
recall_curve.plot()

# Found to perform better than random forest :D

# SVM
svm_ac =[]
svm_re =[]
from sklearn.svm import SVC
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernel:
    svm = SVC(kernel=k)
    svm.fit(x_train, y_train)
    svm_pred = svm.predict(x_test)
    print('For kernel type :', k)
    svm_cm = confusion_matrix(y_test, svm_pred)
    acc_svm = accuracy_score(y_test, svm_pred)
    print('Accuracy for SVM model  is: ', acc_svm)
    recall_svm = recall_score(y_test, svm_pred)
    print('Recall for SVM model  is: ', recall_svm)
    print(svm_cm)
    svm_ac.append(acc_svm)
    svm_re.append(recall_svm)

# linear seems like the best kernel

from sklearn.naive_bayes import GaussianNB
gm = GaussianNB()
gm.fit(x_train, y_train)
gm_pred = gm.predict(x_test)
gm_cm = confusion_matrix(y_test, gm_pred)
acc_gm = accuracy_score(y_test, gm_pred)
print('Accuracy for Naive Bayes model  is: ', acc_gm)
recall_gm = recall_score(y_test, gm_pred)
print('Recall for Naive Bayes model  is: ', recall_gm)
print(gm_cm)

