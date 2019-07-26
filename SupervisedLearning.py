#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Analysis and Cleaning

# In[ ]:


df = pd.read_excel('default_of_credit_card_clients.xls', index_col="ID", skiprows=[0])

print(df.head())
print(df.describe())
print(df.info())

# One Hot-Coding for categorical features : binary features take values of 1 or 0
# - Scikit-learn might assume these are numerical features
# - can't use labels because Scikit-learn only accepts numbers

# obtain the one hot encoding of columns 'SEX', 'EDUCATION', 'MARRIAGE'
# The base values are: female, other_education, other_marital_status
df['male'] = (df['SEX'] == 1).astype('int')
df.drop('SEX', axis=1, inplace=True)

df['grad_school'] = (df['EDUCATION'] == 1).astype('int')
df['university'] = (df['EDUCATION'] == 2).astype('int')
df['high_school'] = (df['EDUCATION'] == 3).astype('int')
df.drop('EDUCATION', axis=1, inplace=True)

df['married'] = (df['MARRIAGE'] == 1).astype('int')
df['single'] = (df['MARRIAGE'] == 2).astype('int')
df.drop('MARRIAGE', axis=1, inplace=True)

# From the documentation, we can infer that PAY_n features represent not delayed if it is <= 0

# modify all values of PAY_n features which are < 0 to 0
pay_n_features = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
for pay_n in pay_n_features:
    df.loc[df[pay_n] <= 0, pay_n] = 0

df.rename(columns={'default payment next month': 'default'}, inplace=True)

df


# ## Building Machine Learning Models

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler


# In[ ]:


# Feature scaling to get more accurate representation and better learning performance
'''
Most machine learning algorithms take into account only the magnitude of the measurements, not the units of those measurements.
The feature with a very high magnitude (number) may affect the prediction a lot more than an equally important feature.
e.g. the AGE (within certain fixed range) and the PAY_AMTn (monetary) features have very different ranges of values

RobustScaler:
The Robust Scaler uses statistics that are robust to outliers.
This usage of interquartiles means that they focus on the parts where the bulk of the data is.
This makes them very suitable for working with outliers.
Notice that after Robust scaling, the distributions are brought into the same scale and overlap, but the outliers remain outside of bulk of the new distributions.
'''
x = df.drop('default', axis=1)
robust_scaler = RobustScaler()
x = robust_scaler.fit_transform(x)# rescale all the features to a same range
y = df['default']
# stratify parameter makes data split in a stratified fashion meaning the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=123, stratify=y)


# In[ ]:


def c_matrix(CM, labels=['pay', 'default']):
    df = pd.DataFrame(data = CM, index=labels, columns=labels)
    df.index.name = 'TRUE'
    df.columns.name = 'PREDICTION'
    df.loc['Total'] = df.sum()
    df['Total'] = df.sum(axis=1)
    return df


# ## Evaluating Model Performance

# In[ ]:


# Preparing dataframe to store the evaluation metrics
metrics = pd.DataFrame(
    index=['accuracy', 'precision', 'recall'],
    columns=['NULL', 'LogisticReg', 'DecisionTree', 'NaiveBayes']
)


# ### In this application
# 
# 1. Accuracy: Overall how often the model predicts correctly defaulters and non-defaulters?
# 2. Precision: When the model predicts defaults: how often is correct?
# 3. Recall: The proportion of actual defaulters that the model will correctly predict?
# 
# ### Which metric to use?
# 1. False positive: A person who will pay predicted as defaulter
# 2. False negative: A person who will default predicted as payer
# 
# #### False negatives are worse => look for better recall
# 
# ## The Null model: always predict the most common category

# In[ ]:


# benchmark or base for how good the model must be performed to beat the Null model
# predict the most common category which is 'pay'
y_predicted = np.repeat(y_train.value_counts().idxmax(), y_test.size)
metrics.loc['accuracy', 'NULL'] = accuracy_score(y_pred=y_predicted, y_true=y_test)
metrics.loc['precision', 'NULL'] = precision_score(y_pred=y_predicted, y_true=y_test)
metrics.loc['recall', 'NULL'] = recall_score(y_pred=y_predicted, y_true=y_test)

# construct the confusion matrix
CM = confusion_matrix(y_pred=y_predicted, y_true=y_test)
c_matrix(CM)


# ## 1. Logistic Regression

# In[ ]:


# import the model class
from sklearn.linear_model import LogisticRegression

# create an instance of the model
log_reg = LogisticRegression(n_jobs=-1, random_state=15)

# train the model using the training data
log_reg.fit(x_train, y_train)

# evaluate the model performance
y_predicted = log_reg.predict(x_test)
metrics.loc['accuracy', 'LogisticReg'] = accuracy_score(y_pred=y_predicted, y_true=y_test)
metrics.loc['precision', 'LogisticReg'] = precision_score(y_pred=y_predicted, y_true=y_test)
metrics.loc['recall', 'LogisticReg'] = recall_score(y_pred=y_predicted, y_true=y_test)

# construct the confusion matrix
CM = confusion_matrix(y_pred=y_predicted, y_true=y_test)
c_matrix(CM)


# ## 2. Decision Tree Classifier

# In[ ]:


# import the model class
from sklearn.tree import DecisionTreeClassifier

# create an instance of the model
'''
min_samples_split => minimum number of samples required to split an internal node
min_samples_leaf => minimum number of samples required to be at a leaf node
'''
dec_tree = DecisionTreeClassifier(min_samples_split=30, min_samples_leaf=10, random_state=10)

# train the model using the training data
dec_tree.fit(x_train, y_train)

# evaluate the model performance
y_predicted = dec_tree.predict(x_test)
metrics.loc['accuracy', 'DecisionTree'] = accuracy_score(y_pred=y_predicted, y_true=y_test)
metrics.loc['precision', 'DecisionTree'] = precision_score(y_pred=y_predicted, y_true=y_test)
metrics.loc['recall', 'DecisionTree'] = recall_score(y_pred=y_predicted, y_true=y_test)

# construct the confusion matrix
CM = confusion_matrix(y_pred=y_predicted, y_true=y_test)
c_matrix(CM)


# ## 3. Naive Bayes Classifier

# In[ ]:


# import the model class
from sklearn.naive_bayes import GaussianNB# for features with continuous values

# create an instance of the model
nb_classifier = GaussianNB()

# train the model using the training data
nb_classifier.fit(x_train, y_train)

# evaluate the model performance
y_predicted = nb_classifier.predict(x_test)
metrics.loc['accuracy', 'NaiveBayes'] = accuracy_score(y_pred=y_predicted, y_true=y_test)
metrics.loc['precision', 'NaiveBayes'] = precision_score(y_pred=y_predicted, y_true=y_test)
metrics.loc['recall', 'NaiveBayes'] = recall_score(y_pred=y_predicted, y_true=y_test)

# construct the confusion matrix
CM = confusion_matrix(y_pred=y_predicted, y_true=y_test)
c_matrix(CM)


# ## Metrics Analysis and Visualization

# In[ ]:


100 * metrics


# In[ ]:


fig, ax = plt.subplots(figsize=(8,5))
metrics.plot(kind='barh', ax=ax)
ax.grid()


# In[ ]:


# adjust precision and recall by modifying the classification thresholds
# predict_proba gives you the probabilities for the target (0 and 1 in your case) in array form
precision_nb, recall_nb, thresholds_nb = precision_recall_curve(y_true=y_test, probas_pred=nb_classifier.predict_proba(x_test)[:,1])

precision_lr, recall_lr, thresholds_lr = precision_recall_curve(y_true=y_test, probas_pred=log_reg.predict_proba(x_test)[:,1])


# In[ ]:


fig, ax = plt.subplots(figsize=(8,5))
ax.plot(precision_nb, recall_nb, label='NaiveBayes')
ax.plot(precision_lr, recall_lr, label='LogisticReg')
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_title('Precision-Recall Curve')
ax.hlines(y=0.5, xmin=0, xmax=1, color='red')
ax.legend()
ax.grid()

# Logistic regression is better than Naive Bayes


# ## Confusion Matrix for modified Logistic Regression Classifier

# In[ ]:


fig, ax = plt.subplots(figsize=(8,5))
print(thresholds_lr)
print(precision_lr)
ax.plot(thresholds_lr, precision_lr[1:], label='Precision')
ax.plot(thresholds_lr, recall_lr[1:], label='Recall')
ax.set_xlabel('Classfication Threshold')
ax.set_ylabel('Precision, Recall')
ax.set_title('Logistic Regression Classifier: Precision-Recall')
ax.hlines(y=0.6, xmin=0, xmax=1, color='red')
ax.legend()
ax.grid()


# ## Classifier with threshold of 0.2

# In[ ]:


y_pred_proba = log_reg.predict_proba(x_test)[:,1]
y_predicted = (y_pred_proba >= 0.2).astype('int')
# adjust the original classification threshold from 0.5 to 0.2

# confusion matrix
CM = confusion_matrix(y_pred=y_predicted, y_true=y_test)
print("Recall: ", 100*recall_score(y_pred=y_predicted, y_true=y_test))
print("Precision: ", 100*precision_score(y_pred=y_predicted, y_true=y_test))
c_matrix(CM)


# ## Final Predictive Model (Logistic Regression)

# In[ ]:


def predict_default(new_data):
    '''
    #print(new_data)
    #print(new_data.shape)
    # The criterion to satisfy for providing the new shape is that 'The new shape should be compatible with the original shape'
    # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
    '''
    data = new_data.values.reshape(1, -1)
    data = robust_scaler.transform(data)
    prob = log_reg.predict_proba(data)[0][1]
    if prob >= 0.2:
        return "Will default"
    else:
        return "Will pay"


# In[ ]:


pay = df[df['default']==0]


# In[ ]:


pay.head()


# In[ ]:


from collections import OrderedDict
new_customer = OrderedDict([
    ('LIMIT_BAL', 4000), ('AGE', 50), ('BILL_AMT1', 500),
    ('BILL_AMT2', 35509), ('BILL_AMT3', 689), ('BILL_AMT4', 0),
    ('BILL_AMT5', 0), ('BILL_AMT6', 0), ('PAY_AMT1', 0),
    ('PAY_AMT2', 35509), ('PAY_AMT3', 0), ('PAY_AMT4', 0),
    ('PAY_AMT5', 0), ('PAY_AMT6', 0), ('male', 1), ('grad_school', 0),
    ('university', 1), ('high_school', 0), ('married', 1), ('single', 0), ('pay_0', -1),
    ('pay_2', -1), ('pay_3', -1), ('pay_4', 0), ('pay_5', -1), ('pay_6', 0),
])

new_customer = pd.Series(new_customer)
predict_default(new_customer)


# In[ ]:


'''
for x in negative.index[0:100]:
    print(predict_default(negative.loc[x].drop('default')))
'''


# In[ ]:




