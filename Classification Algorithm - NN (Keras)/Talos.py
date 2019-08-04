#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123, stratify=y)


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
    columns=['NULL', 'LogisticReg', 'DecisionTree', 'NaiveBayes', 'NeuralNet']
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


# ## 4. Feed Forward Deep Neural Networks

# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

input_dim = x_train.shape[1]
print(input_dim)

model = Sequential()
model.add(Dense(256, input_shape=(input_dim,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1,  activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.callbacks import Callback

class BatchLogger(Callback):
    def on_train_begin(self, epoch, logs={}):
        self.log_values = {}
        for k in self.params['metrics']:
            self.log_values[k] = []

    def on_epoch_end(self, batch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values[k].append(logs[k])
    
    def get_values(self, metric_name, window):
        d =  pd.Series(self.log_values[metric_name])
        return d.rolling(window,center=False).mean()

bl = BatchLogger()

history = model.fit(np.array(x_train), np.array(y_train),
              batch_size=25, epochs=10, verbose=1, callbacks=[bl],
              validation_data=(np.array(x_test), np.array(y_test)))


# In[ ]:


score = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
print('Test log loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.title('loss, per batch')
plt.plot(bl.get_values('loss',1), 'b-', label='train');
plt.plot(bl.get_values('val_loss',1), 'r-', label='test');

plt.subplot(1, 2, 2)
plt.title('accuracy, per batch')
plt.plot(bl.get_values('acc',1), 'b-', label='train');
plt.plot(bl.get_values('val_acc',1), 'r-', label='test');
plt.show()


# In[ ]:


import itertools
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, accuracy_score, confusion_matrix

def plot_cm(ax, y_true, y_pred, classes, title, th=0.5, cmap=plt.cm.Blues):
    y_pred_labels = (y_pred>th).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_labels)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

def plot_auc(ax, y_train, y_train_pred, y_test, y_test_pred, th=0.5):

    y_train_pred_labels = (y_train_pred>th).astype(int)
    y_test_pred_labels  = (y_test_pred>th).astype(int)

    fpr_train, tpr_train, _ = roc_curve(y_train,y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    acc_train = accuracy_score(y_train, y_train_pred_labels)

    fpr_test, tpr_test, _ = roc_curve(y_test,y_test_pred)
    roc_auc_test = auc(fpr_test, tpr_test)
    acc_test = accuracy_score(y_test, y_test_pred_labels)

    ax.plot(fpr_train, tpr_train)
    ax.plot(fpr_test, tpr_test)

    ax.plot([0, 1], [0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    
    train_text = 'train acc = {:.3f}, auc = {:.2f}'.format(acc_train, roc_auc_train)
    test_text = 'test acc = {:.3f}, auc = {:.2f}'.format(acc_test, roc_auc_test)
    ax.legend([train_text, test_text])


# In[ ]:


y_train_pred = model.predict_on_batch(np.array(x_train))[:,0]
y_test_pred = model.predict_on_batch(np.array(x_test))[:,0]

fig,ax = plt.subplots(1,3)
fig.set_size_inches(15,5)

plot_cm(ax[0], y_train, y_train_pred, [0,1], 'Confusion matrix (TRAIN)')
plot_cm(ax[1], y_test, y_test_pred, [0,1], 'Confusion matrix (TEST)')

plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred)
    
plt.tight_layout()
plt.show()


# ## <font color=red>Hyperparameter Tuning for Sequential Model (Using GridSearchCV)</font>

# #### To use Keras model in Scikit Learn, we need to use the KerasClassifier or KerasRegressor classes. These two classes accept a function which creates and returns a Keras model. 

# In[ ]:


############################################################## Talos ##############################################################
# define the function for Sequential model
def credit_card_default_model(x_train, y_train, x_val, y_val, params):

    # build the model
    model = Sequential()
    
    # construct the layers (input, hidden, output)
    model.add(Dense(params['first_neuron'], input_dim=x_train.shape[1],
                    activation=params['activation'],
                    kernel_initializer=params['kernel_initializer']))
    
    model.add(Dropout(params['dropout']))
    
    model.add(Dense(params['second_neuron'],
                    activation=params['activation'],
                    kernel_initializer=params['kernel_initializer']))
   
    model.add(Dropout(params['dropout']))

    model.add(Dense(params['third_neuron'],
                    activation=params['activation'],
                    kernel_initializer=params['kernel_initializer']))
    
    model.add(Dropout(params['dropout']))
    
    model.add(Dense(params['fourth_neuron'],
                    activation=params['activation'],
                    kernel_initializer=params['kernel_initializer']))
    
    model.add(Dropout(params['dropout']))
    
    model.add(Dense(params['fifth_neuron'],
                    activation=params['activation'],
                    kernel_initializer=params['kernel_initializer']))
    
    model.add(Dropout(params['dropout']))

    model.add(Dense(1, activation=params['last_activation'],
                    kernel_initializer=params['kernel_initializer']))
    
    # compile the model
    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'](),
                  metrics=['acc', fmeasure_acc])
    
    # train the model
    history = model.fit(x_train, y_train, 
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        callbacks=[live()],
                        epochs=params['epochs'],
                        verbose=0)

    return history, model


# In[ ]:


import talos as ta
from talos.metrics.keras_metrics import fmeasure_acc
from talos import live

import pandas as pd

from keras.models import Sequential
from keras.layers import Dropout, Dense

# Keras items
from keras.optimizers import Adam, Nadam
from keras.activations import relu, elu
from keras.losses import binary_crossentropy
from keras.wrappers.scikit_learn import KerasClassifier

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# then we can go ahead and set the parameter space
p = {
    'first_neuron':[9,10,11],
     'second_neuron':[32,64],
     'third_neuron':[32,64],
     'fourth_neuron':[64,128],
     'fifth_neuron':[64,128],
     'hidden_layers':[0, 1, 2],
     'batch_size': [30],
     'epochs': [100],
     'dropout': [0],
     'kernel_initializer': ['uniform','normal'],
     'optimizer': [Nadam, Adam],
     'losses': [binary_crossentropy],
     'activation':[relu, elu],
     'last_activation': ['sigmoid', 'softmax']
}


# In[ ]:


# run the experiment
t = ta.Scan(x=x,
            y=y,
            model=credit_card_default_model,
            params=p,
            dataset_name='Credit Card Default',
            experiment_no='1')


# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

input_dim = x_train.shape[1]
print(x_train.shape)
print(input_dim)

model = Sequential()
model.add(Dense(64, input_shape=(input_dim,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1,  activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.callbacks import Callback

class BatchLogger(Callback):
    def on_train_begin(self, epoch, logs={}):
        self.log_values = {}
        for k in self.params['metrics']:
            self.log_values[k] = []

    def on_epoch_end(self, batch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values[k].append(logs[k])
    
    def get_values(self, metric_name, window):
        d =  pd.Series(self.log_values[metric_name])
        return d.rolling(window,center=False).mean()

bl = BatchLogger()

history = model.fit(np.array(x_train), np.array(y_train),
              batch_size=32, epochs=20, verbose=1, callbacks=[bl],
              validation_data=(np.array(x_test), np.array(y_test)))


# In[ ]:


score = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
print('Test log loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.title('loss, per batch')
plt.plot(bl.get_values('loss',1), 'b-', label='train');
plt.plot(bl.get_values('val_loss',1), 'r-', label='test');

plt.subplot(1, 2, 2)
plt.title('accuracy, per batch')
plt.plot(bl.get_values('acc',1), 'b-', label='train');
plt.plot(bl.get_values('val_acc',1), 'r-', label='test');
plt.show()


# In[ ]:


y_train_pred = model.predict_on_batch(np.array(x_train))[:,0]
y_test_pred = model.predict_on_batch(np.array(x_test))[:,0]

fig,ax = plt.subplots(1,3)
fig.set_size_inches(15,5)

plot_cm(ax[0], y_train, y_train_pred, [0,1], 'Confusion matrix (TRAIN)')
plot_cm(ax[1], y_test, y_test_pred, [0,1], 'Confusion matrix (TEST)')

plot_auc(ax[2], y_train, y_train_pred, y_test, y_test_pred)
    
plt.tight_layout()
plt.show()

############################################################## Talos ##############################################################

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




