#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering
# 
# #### MDS Quality Measures Dataset 
# - [Import Necessary Modules and Dataset](#import)
# 
# #### Provider Information Dataset
# - [Import Dataset and Quick Assessment](#import1)
# - [Training, Validation, and Testing Split](#split)
# - [Standardize Features](#standardize)
# - [Featuring Selection](#feature)
# - [Data Leakage Investigation](#leakage)

# ## MDS Quality Measures Dataset 

# ### Import Necessary Modules and Dataset <a id = 'import'></a>

# In[1]:


# Import tools to get datasets
from io import BytesIO
from zipfile import ZipFile
import urllib

# Import data manipulation and plot modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Import ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.feature_selection import VarianceThreshold


# In[2]:


# Unzip and read Q
url = urllib.request.urlopen('https://github.com/jonahwinninghoff/Springboard_Capstone_Project/blob/main/Assets/cleaned_quality.csv.zip?raw=true')
file = ZipFile(BytesIO(url.read()))
qfile = file.open("cleaned_quality.csv")
q = pd.read_csv(qfile, encoding='cp1252').drop('Unnamed: 0', axis = 1)
file.close()


# In[3]:


display(q.head())


# In[4]:


q.info()


# In[5]:


q = q.drop(['Short Stay', 'Used in Quality Meaure Five Star Rating', 'Provider Zip Code',
           'Provider Name', 'Provider Address', 'Provider City', 'Location'], axis = 1)


# In[6]:


display(q.describe())


# In[7]:


q.to_csv('for_ANOVA_analysis')


# This dataset contains dummy variables only, so no further action is required.

# ## Provider Information Dataset

# ### Import Dataset and Quick Assessment <a id = 'import1'></a>

# In[8]:


# Unzip and read Provider Info
url = urllib.request.urlopen('https://github.com/jonahwinninghoff/Springboard_Capstone_Project/blob/main/Assets/cleaned_information.csv.zip?raw=true')
file = ZipFile(BytesIO(url.read()))
qfile = file.open("cleaned_information.csv")
q = pd.read_csv(qfile, encoding='cp1252').drop('Unnamed: 0', axis = 1)
file.close()


# In[9]:


display(q.head())


# In[10]:


display(q.info())


# In[11]:


# Ownership Type should be dummies, so fix it
q = pd.concat([q.drop('Ownership Type',axis=1),pd.get_dummies(q['Ownership Type'],drop_first = True)],axis=1)


# In[12]:


display(q.describe())


# In[13]:


counter = Counter(q['Provider State'])
counter = dict(sorted(counter.items(), key=lambda kv: kv[1]))
plt.style.use('fivethirtyeight')
plt.figure(figsize = (15,7))
plt.bar(counter.keys(),counter.values())
plt.title('Frequency of States Appeared in Dataset', family = 'monospace')
plt.xticks(rotation = 90)
plt.grid(False)
plt.show()


# In[14]:


q['National Area regional code'] = np.trunc(q['Provider Zip Code']/10000).astype('int')


# In[15]:


q=q.select_dtypes(exclude='object')


# ### Training, Validation, Testing Split <a id = 'split'></a>

# In[16]:


# Split test and train, and separate X, Overall Rating y, and Abuse Icon y
X_train, X_test, abu_y_train, abu_y_test = train_test_split(q.drop(['Overall Rating', 'Abuse Icon'],axis =1),
                                                   q['Abuse Icon'],
                                                   test_size=.40, random_state = 111)

X_train1, X_test1, rat_y_train, rat_y_test = train_test_split(q.drop(['Overall Rating', 'Abuse Icon'],axis =1),
                                                   q['Overall Rating'],
                                                   test_size=.40, random_state = 111)


# In[17]:


# Split test and valid for Abuse Icon and Overall Rating
X_valid, X_test, abu_y_valid, abu_y_test = train_test_split(X_test,
                                                   abu_y_test,
                                                   test_size=.5, random_state = 111)

X_valid1, X_test1, rat_y_valid, rat_y_test = train_test_split(X_test1,
                                                   rat_y_test,
                                                   test_size=.5, random_state = 111)


# In[18]:


print({'training size':len(X_train),
       'testing size':len(X_test),
       'validation size':len(X_valid)})


# ### Standardize Features <a id = 'standardize'></a>

# In[19]:


# Standardize all sets without data leakage
zscore = StandardScaler()
zscore.fit(X_train)

zscore1 = StandardScaler()
zscore1.fit(X_train1)

# Apply it
X_train = pd.DataFrame(zscore.transform(X_train))
X_valid = pd.DataFrame(zscore.transform(X_valid))
X_test = pd.DataFrame(zscore.transform(X_test))

X_train1 = pd.DataFrame(zscore1.transform(X_train1))
X_valid1 = pd.DataFrame(zscore1.transform(X_valid1))
X_test1 = pd.DataFrame(zscore1.transform(X_test1))

# Rename columns
X_train.columns = list(q.drop(['Overall Rating','Abuse Icon'],axis=1).columns)
X_valid.columns = list(q.drop(['Overall Rating','Abuse Icon'],axis=1).columns)
X_test.columns = list(q.drop(['Overall Rating','Abuse Icon'],axis=1).columns) 

X_train1.columns = list(q.drop(['Overall Rating','Abuse Icon'],axis=1).columns)
X_valid1.columns = list(q.drop(['Overall Rating','Abuse Icon'],axis=1).columns)
X_test1.columns = list(q.drop(['Overall Rating','Abuse Icon'],axis=1).columns)


# ### Feature Selection <a id = 'feature'></a>

# In[20]:


lm = LinearRegression()
lm.fit(X_train1,rat_y_train)


# In[21]:


print('Training R2:',lm.score(X_train1,rat_y_train))
print('Valid R2:',lm.score(X_valid1,rat_y_valid))


# BIC and AIC scores are too high.

# In[22]:


# Set the threshold to eliminate high correlation in X
def eliminate_X_corr(X, threshold):
    # Correlate using Pearson
    data = pd.DataFrame(X)
    correlated = pd.DataFrame(data.corr().unstack().sort_values())
    
    # Drop all pair of identical variables being correlated
    correlated.drop(correlated.tail(len(data.select_dtypes(include = ['float',
                    'integer']).columns)).index,inplace=True)
    
    # Limit correlated variables that are higher than threshold or lower than - threshold
    correlated = correlated[(correlated > threshold)]
    correlated = correlated[~correlated[0].isna()].reset_index()
    
    # Drop duplications
    thelist = []
    for i in range(len(correlated)):
        if i % 2 == 0:
            thelist.append(i)
    correlated = correlated.iloc[thelist,:]
    
    # Select list of columns to drop
    thelist = list(correlated.iloc[:,0])
    return data.drop(thelist,axis=1)


# In[23]:


sequence = []
i = 0.2
while i <= 1:
    sequence.append(i)
    i += 0.01
    i = round(i,2)

# Create the graph
train = []
valid = []
for i in sequence:
    X = eliminate_X_corr(X_train1,i)
    lm.fit(X,rat_y_train)
    
    X_val = X_valid1[X.columns]
    
    train.append(lm.score(X,rat_y_train))
    valid.append(lm.score(X_val,rat_y_valid))


# In[24]:


plt.figure(figsize=(15,7))
plt.plot(train,label='training set')
plt.plot(valid,label='validation set')
plt.legend()
plt.show()


# In[25]:


log = LogisticRegression()
log.fit(X_train,abu_y_train)


# In[26]:


print('Training R2:', matthews_corrcoef(abu_y_train, log.predict(X_train)))
print('Valid R2:',matthews_corrcoef(abu_y_valid, log.predict(X_valid)))


# In[27]:


sequence = []
i = 0.2
while i <= 1:
    sequence.append(i)
    i += 0.01
    i = round(i,2)

# Create the graph
train = []
valid = []
for i in sequence:
    X = eliminate_X_corr(X_train,i)
    log.fit(X,abu_y_train)
    
    X_val = X_valid[X.columns]
    
    train.append(matthews_corrcoef(abu_y_train, log.predict(X)))
    valid.append(matthews_corrcoef(abu_y_valid, log.predict(X_val)))


# In[28]:


plt.figure(figsize=(15,7))
plt.plot(train,label='training set')
plt.plot(valid,label='validation set')
plt.legend()
plt.show()


# In[29]:


sequence = []
i = 0.0
while i <= 1:
    sequence.append(i)
    i += 0.01
    i = round(i,2)

# Create the graph
train = []
valid = []
thei = []
train1 = []
valid1 = []
for i in sequence:
    sel = VarianceThreshold(threshold = i)
    sel.fit(X_train) 
    X = sel.transform(X_train)
    log.fit(X,abu_y_train)
    
    X_val = sel.transform(X_valid)
    
    train.append(matthews_corrcoef(abu_y_train, log.predict(X)))
    valid.append(matthews_corrcoef(abu_y_valid, log.predict(X_val)))

    sel1 = VarianceThreshold(threshold = i)
    sel1.fit(X_train1)
    X1 = sel1.transform(X_train1)
    
    lm.fit(X1,rat_y_train)
    
    X_val1 = sel1.transform(X_valid1)
    
    train1.append(lm.score(X1,rat_y_train))
    valid1.append(lm.score(X_val1,rat_y_valid))
    thei.append(i)


# In[30]:


scores = pd.DataFrame({'threshold':thei,'classifer_training':train,'regressor_training':train1,
                       'classifer_validation':valid,'regressor_validation':valid1})


# In[31]:


plt.figure(figsize=(15,7))
plt.plot('threshold','classifer_training',data=scores,label='training set')
plt.plot('threshold','classifer_validation',data=scores,label='validation set')
plt.title('Abuse Icon Classifer',family='Monospace')
plt.legend()
plt.show()


# In[32]:


plt.figure(figsize=(15,7))
plt.plot('threshold','regressor_training',data=scores,label='training set')
plt.plot('threshold','regressor_validation',data=scores,label='validation set')
plt.title('Abuse Icon Classifer',family='Monospace')
plt.legend()
plt.show()


# ### Data Leakage Investigation <a id = 'leakage'></a>

# In[33]:


# Remove all ratings and inspection-related inputs other than Health Inspection Rating
eliminate = []
for i in q.columns:
    if ('Rating' in i) and ('Inspection Rating' not in i):
        eliminate.append(i)

eliminate.extend(['Number of Citations from Infection Control Inspections',
                  'Most Recent Health Inspection More Than 2 Years Ago',
                  'Total Weighted Health Survey Score'])
q = q.drop(eliminate,axis=1)


# In[34]:


q.to_csv('for_prediction')

