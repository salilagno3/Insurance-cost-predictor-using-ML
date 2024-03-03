#!/usr/bin/env python
# coding: utf-8

# # Importing the Dependencies 
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


# # Data Collection and Analysis
# 

# In[2]:


#loading the data from csv file to a Pandas DataFrame
df=pd.read_csv('C:\\Users\\Salil\\Downloads\\insurance.csv')


# In[3]:


#number of rows and columns
df.shape


# In[4]:


#Data of the DataFrame
df.head


# In[5]:


#information about the dataset
df.info()


# In[6]:


#checking for missing values
df.isnull().sum()


# # Data Analysis
# 

# In[7]:


#statistical measures of the dataset
df.describe()


# In[8]:


#distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(df['age'])
plt.title('Age Distribution')
plt.show()


# In[9]:


#Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=df)
plt.title('Sex Distribution')
plt.show()


# In[10]:


#Gender count
df['sex'].value_counts()


# In[11]:


#Bmi column
plt.figure(figsize=(6,6))
sns.distplot(df['bmi'])
plt.title('BMI Distribution')
plt.show()
#normal BMI range -> 18.5-24.9


# In[12]:


#children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=df)
plt.title('Number of children')
plt.show()


# In[13]:


df['children'].value_counts()


# In[14]:


#smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=df)
plt.title('Smokers')
plt.show()


# In[15]:


df['smoker'].value_counts()


# In[16]:


#region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=df)
plt.title('Regions')
plt.show()


# In[17]:


df['region'].value_counts()


# In[18]:


#dsitribution of charges 
plt.figure(figsize=(6,6))
sns.distplot(df['charges'])
plt.title('Charges Distribution')
plt.show()


# # Data Pre-processing

# # Encoding categorical features

# In[19]:


#encoding sex column
df.replace({'sex':{'male':0, 'female':1}}, inplace=True)

#encoding smoker column
df.replace({'smoker':{'yes':0, 'no':1}}, inplace=True)

#encoding region column
df.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}}, inplace=True )


# # Splitting Features and Target

# In[20]:


X=df.drop(columns='charges', axis=1)
Y=df['charges']


# In[21]:


print(X)


# In[22]:


print(Y)


# # Splitting Data into Training Data and Testing Data

# In[23]:


X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=2 )


# In[24]:


print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)


# # Model Training
# 

# # Linear Regression Model
# 

# In[25]:


#loading the model
regressor= LinearRegression()


# In[26]:


regressor.fit(X_train, Y_train)


# # Model Evaluation
# 

# In[27]:


from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt 
from sklearn.model_selection import cross_val_predict  
def model_summary(model, model_name, cvn=20): # Default value for cvn = 20
    print(model_name)
    y_pred_model_train = model.predict(X_train)
    y_pred_model_test = model.predict(X_test)
    R2Score_model_train = r2_score(Y_train, y_pred_model_train)
    print("Training R2 Score: ", R2Score_model_train)
    R2Score_model_test = r2_score(Y_test, y_pred_model_test)
    print("Testing R2 Score: ",  R2Score_model_test)
    RMSE_model_train = sqrt(mean_squared_error(Y_train, y_pred_model_train))
    print("RMSE for Training Data: ", RMSE_model_train)
    RMSE_model_test = sqrt(mean_squared_error(Y_test, y_pred_model_test))
    print("RMSE for Testing Data: ", RMSE_model_test)
    y_pred_cv_model = cross_val_predict(model, X, Y, cv=cvn)
    accuracy_cv_model = r2_score(Y, y_pred_cv_model)
    print("Accuracy for", cvn,"- Fold Cross Predicted: ", accuracy_cv_model)


# In[28]:


#prediction on data
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
model_summary(regressor, "Multiple_linear_Regression")


# In[29]:


from sklearn.tree import DecisionTreeRegressor
decision_tree_reg = DecisionTreeRegressor(max_depth=5, random_state=13)  
decision_tree_reg.fit(X_train, Y_train) 
model_summary(decision_tree_reg, "Decision_Tree_Regression")


# In[30]:


from sklearn.ensemble import RandomForestRegressor
random_forest_reg=RandomForestRegressor()
random_forest_reg.fit(X_train,Y_train)
model_summary(random_forest_reg,"Random_Forest_Regressor")


# In[32]:


from sklearn.ensemble import GradientBoostingRegressor
reg_grad=GradientBoostingRegressor()
reg_grad.fit(X_train, Y_train)
model_summary(reg_grad, "Gradient Boosting")


# In[33]:


from xgboost import XGBRegressor
reg_xg=XGBRegressor()
reg_xg.fit(X_train, Y_train)
model_summary(reg_xg, "XGBRegressor")


# # Building a Predictive system
# 

# In[34]:


age= int(input("Enter your age :"))
sex= int(input("Enter 0 for male and 1 for female :"))
bmi= float(input("Enter BMI :"))
chi= int(input("Enter number of children :"))
smo= int(input("Enter 0 if smoker, enter 1 if not :"))
reg= int(input("Enter 0 for southeast, 1 for southwest, 2 for northeast, 3 for northwest :"))

inp = (age, sex, bmi, chi, smo, reg)
 
#changing input data to numpy array
inp_data = np.asarray(inp)

#reshaping array
inp_data_res = inp_data.reshape(1,-1)

pred = reg_grad.predict(inp_data_res)
print('Predicted insurance cost : USD',pred[0])


# In[ ]:




