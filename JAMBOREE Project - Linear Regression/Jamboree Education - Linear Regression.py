#!/usr/bin/env python
# coding: utf-8

# Jamboree has helped thousands of students like you make it to top colleges abroad. Be it GMAT, GRE or SAT, their unique problem-solving methods ensure maximum scores with minimum effort.
# They recently launched a feature where students/learners can come to their website and check their probability of getting into the IVY league college. This feature estimates the chances of graduate admission from an Indian perspective.

# # Problem Statement :

# - Our Analysis will help Jamboree in understanding what factors are important in graduate admissions and how these factors are interrelated among themselves. 
# - Our Analysis will also help predict one's chances of admission given the rest of the variables.

# # Data Description :

# Jamboree collected the Data of different Students who enrolled in Jamboree for preparation in GMAT, GRE or SAT
# Exams. The Dataset has the following features:
#     
# - Serial Number - Represents the Unique Row ID for each Student
# - GRE Scores - GRE Scores obtained out of 340
# - TOEFL Scores - TOEFL Scores obtained out of 120
# - University Ranking - Rankings of Different Universities from 0 to 5
# - Statement of Purpose and Letter of Recommendation Strength - SOP and LOR ratings out of 5
# - Undergraduate GPA - Marks obtained during the Undergraduation from 0 to 10
# - Research Experience - Whether any Student has Prior Research Experience (0 or 1)
# - Chances of Admit - Whether any Student will get Admission (0 or 1)

# # Importing Required Libraries : 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Train Data and Test Data Split
from sklearn.model_selection import train_test_split

# Linear Regression using Statsmodel
import statsmodels.api as sm

# Checking for VIF (Variance Inflation Factor) in our Model
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Importing SKLEARN libraries for our Model checking
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , r2_score , mean_squared_error , mean_absolute_percentage_error


# In[2]:


df = pd.read_csv("Jamboree_Admission.csv", index_col = False)
df


# # Exploratory Data Analysis :

# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.columns


# # a) Checking for Null and NaN Values in the Dataset : 

# In[6]:


sns.heatmap(df.isnull(), cbar=False)


# In[7]:


sns.heatmap(df.isna(), cbar=False)


# Thus , Dataset has no Null or NaN Values. So we can proceed further.

# # b) Other Descriptions and Details on Dataset : 

# In[8]:


df.describe(include = 'int')


# In[9]:


df.describe(include = 'float')


# In[10]:


df.describe(include = 'all')


# # Observations : 

# - GRE Score ranges from 290 to 340, with Mean of 316.47 and Median of 317. Since Mean is almost Equal to the Median, thus 'GRE Score' has to be normally distributed.
# - TOEFL Score ranges from 92 to 120, with Mean of 107.19 and Median of 107. Since Mean is almost Equal to the Median, thus 'TOEFL Score' has to be normally distributed.
# - Chance of Admit ranges from 0.34 to 0.97, with Mean of 0.72114 and Median of 0.72. Since Mean is almost Equal to the Median, thus 'Chance of Admit' has to be normally distributed.
# - CGPA ranges from 6.80 to 9.92, with Mean of 8.57 and Median of 8.56. Since Mean is almost Equal to the Median, thus Undergraduate 'CGPA' has to be normally distributed.
# - Dataset has no Duplicated Values. Thus , we find that there are 500 , 49 , 29 , 5 , 9 , 9 , 184 , 2 , 61 unique values for the Columns - 'Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research', 'Chance of Admit '.

# In[11]:


df.duplicated().unique()


# In[12]:


for column_name in df:
    print(column_name,':',df[column_name].nunique())


# In[13]:


# Need to remove the Extra Spaces present in the Columns : 'Chance of Admit ' and 'LOR '
df.rename(columns = {'Chance of Admit ' : 'Chance of Admit' , 'LOR ' : 'LOR'} , inplace = True)


# In[14]:


df.columns


# In[15]:


# We can remove the Unwanted Columns like Serial Number
df.drop(columns = 'Serial No.' , inplace = True)


# In[16]:


# df.drop(columns = 'index' , inplace = True)


# In[17]:


df.head()


# In[18]:


# Following are the Categorical Columns present in our Dataset : SOP , LOR , Research , University Rating
# Need to convert these columns into 'Category' Datatype.

categorical_cols = ['SOP' , 'LOR' , 'University Rating' , 'Research']

for i in categorical_cols :
    df[categorical_cols] = df[categorical_cols].astype('category')


# # c) Univariate Analysis and Outlier Treatment: 

# Univariate Analysis is defined as Analysis carried out on only one (“uni”) variable (“variate”) to
# Summarize or Describe the Variables.

# In[19]:


sns.set_style('whitegrid')


# In[20]:


numerical_cols = ["GRE Score" , "TOEFL Score" , "CGPA" , "Chance of Admit"]


# In[21]:


plt.figure(figsize = (20,10))

for i in range(len(numerical_cols)):
    plt.subplot(2,4,i + 1)
    sns.boxplot(df[numerical_cols[i]])


# In[22]:


plt.figure(figsize=(20,10))

for i in range(len(numerical_cols)):
    plt.subplot(2,4,i + 1)
    sns.distplot(df[numerical_cols[i]])
    plt.axvline(df[numerical_cols[i]].mean() , color = 'r' , linestyle = '--' , label = 'Mean')
    plt.axvline(df[numerical_cols[i]].median() , color = 'm' , linestyle = '-' , label = 'Median')
    plt.legend()
plt.show()


# In[23]:


# Checking for Outlier Data present in Different Columns

outliers_in_df = df.copy(deep = True)

for i in range(len(numerical_cols)):
    Q1 = df[numerical_cols[i]].quantile(0.25)
    Q3 = df[numerical_cols[i]].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = round(Q1 - 1.5 * IQR , 2)
    upper_limit = round(Q3 + 1.5 * IQR , 2)
    print(numerical_cols[i])
    print('Lower Limit for Boxplot - ', lower_limit , '\nUpper Limit for Boxplot -', upper_limit)
    outlier_df = df[(df[numerical_cols[i]] > upper_limit) | (df[numerical_cols[i]] < lower_limit)]
    outliers_in_df = outliers_in_df[(outliers_in_df[numerical_cols[i]] <= upper_limit) & (outliers_in_df[numerical_cols[i]] >= lower_limit)]
    
    outlier_percentage = round((len(outlier_df) / len(df)) * 100 , 2)
    print('Total Number of Outliers : ', len(outlier_df))
    print('Percentage of Outlier Records :' , outlier_percentage , '%\n')
    print(outlier_df)
    print()
    
overall_outliers_percentage_df = round(((len(df) - len(outliers_in_df)) / len(df)) * 100 , 2)
print('Total Number of Outliers from all Records :', len(df) - len(outliers_in_df))
print('Percentage of Rows as Outliers :', overall_outliers_percentage_df)


# In[24]:


df = outliers_in_df.copy(deep = True)


# In[25]:


sns.countplot(data = df , y = "University Rating")


# In[26]:


sns.countplot(data = df , x = "SOP")


# In[27]:


sns.countplot(data = df , x = "LOR")


# In[28]:


sns.countplot(data = df , x = "Research")


# # Observations : 

# - Considering the University Rating aspect , Maximum Students were enrolled in the Universities having Ratings of 3.
# - Considering the SOP aspect , Maximum Students had a SOP Ratings of 3 and 4.
# - Considering the LOR aspect , Maximum Students had a LOR Rating of 3.
# - Considering the Research aspect , More number of Students had a Prior Research Experience.
# - There are 2 Outliers present only in 'Chance of Admit' column. Total Percentage of Data being an Outlier is 0.4%, checked using IQR Method.

# # d) Bivariate Analysis : 

# Bivariate Analysis is stated to be an Analysis of any concurrent Relation between two Variables or Attributes.

# In[29]:


categorical_cols = ['SOP' , 'LOR' , 'Research' , 'University Rating' ]
sns.set_style('whitegrid')
fig , axis = plt.subplots(nrows = 2 , ncols = 2 , figsize = (20,15))
fig.subplots_adjust(top = 1.2)
count = 0

for i in range(2):
    for j in range(2):
        sns.boxplot(data = df , y = 'Chance of Admit' , x = categorical_cols[count] , ax = axis[i,j])
        axis[i,j].set_title(f"Chance of Admit vs {categorical_cols[count]}")
        count += 1


# In[30]:


num_cols = ["GRE Score" , "TOEFL Score" , "CGPA"]

for i in range(len(num_cols)):
    plt.scatter(df[num_cols[i]] , y = df['Chance of Admit'])
    sns.regplot(x = num_cols[i] , y = 'Chance of Admit' , data = df)
    plt.xlabel(num_cols[i])
    plt.ylabel('Chance of Admit')
    plt.show()


# In[31]:


# Check for Correlation within Different Columns

sns.pairplot(df)


# In[32]:


sns.pairplot(df , hue = 'Research')


# In[33]:


sns.pairplot(df , hue = 'University Rating')


# In[34]:


# Using Heatmap to check the Correlations as well as for Multicollinearity

sns.heatmap(df.corr() , annot = True)


# # Observations : 

# - Numerical Columns like GRE Scores , TOEFL Scores and CGPA are following a Linear Relationship with respect to Chances of Admit.
# - As per the Pairplot for Research , Students considering themselves towards Research Domain have higher Chances of getting Admission, in comparison to those who don't consider Research into consideration.
# - As per the Pairplot for University Ranking , Students require better GRE , TOEFL and CGPA Scores to get admission into better Universities. Considering the Data , Maximum number of Students got admissions in Universities with Ratings 3. And minimum Number of Students got admission in Universities with Ratings 1.
# - As per the Heatmap , columns 'GRE Scores' , 'TOEFL Scores' and 'CGPA' have a high correlation of around 0.8. Thus , as per the Assumptions of Linear Regression, we need to deal with Multicollinearity.
# - All Numerical column(features) are almost normally distributed whereas 'Chance of Admit' is slightly left skewed. Both mean and median co-incides for all the distribution.
# - Column 'Chance of Admit' has a Positive Linear Relation with all of the Numerical and Categorical Features.
# 
# - Chance of Admit vs Research : 
# 
#   Having Research has more Median Value compared to not having Research. Also, Research Students have higher chances than Non-Researchers. Outliers are present only for Research Aspect.
#   
#   
# - Chance of Admit vs University Rating : 
# 
#   As the University Rating increases, the chances of getting admission also Increases, since Rating '5' has the highest Median value followed by 4 , 3 , 2 and 1. Outliers present for most of the Categories.
#   
#   
# - Chance of Admit vs SOP : 
# 
#   As the SOP Rating increases, the chances of getting admission also Increases, since Rating '5' has the highest Median value followed by others. Outliers present for most of the Categories.
#   
#   
# - Chance of Admit vs LOR : 
# 
#   As the LOR Rating increases, the chances of getting admission also Increases, since Rating '5' has the highest Median value followed by others. Outliers present for most of the Categories.

# # Feature Engineering for Model :

# Since the Columns 'GRE Score' , 'TOEFL Score' and 'CGPA' have high correlation , thus we need to drop two of these columns to remove Multicollinearity.
# With Multicollinearity , our Linear Regression model will show Bad Results.
# 
# We can drop 'TOEFL Score' and 'CGPA' Columns

# In[35]:


df_col = ['GRE Score' , 'University Rating' , 'SOP' , 'LOR' , 'Research' , 'Chance of Admit']
new_df = df[df_col]
new_df.shape


# Also, for our Linear Regression Model to work, we need all our Inputs in the form of Numerical Data, instead of Categorical Data.
# Thus converting the Categorical Columns into Numerical Columns.

# In[36]:


cat_col = ['University Rating' , 'SOP' , 'LOR']
dummyCol = pd.get_dummies(new_df[categorical_cols] , drop_first = True)

dummyCol.rename(columns = {'Research_1 ' : 'University Rating_1'} , inplace = True)
dummyCol.head()


# In[37]:


new_df = pd.concat([new_df , dummyCol] , axis = 1)
new_df.drop(cat_col , axis = 1 , inplace = True)

new_df.shape


# In[38]:


# Rescaling using Standard Scaler

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

num_col = ['GRE Score' , 'Chance of Admit']
new_df[num_col] = scaler.fit_transform(new_df[num_col])


# In[39]:


new_df.head()


# # Model Preparation : 

# In[40]:


# Train - Test Split

df_train , df_test = train_test_split(new_df , train_size = 0.7 , random_state = 100)


# In[41]:


df_train.shape


# In[42]:


df_test.shape


# In[43]:


df_train.describe().T


# In[44]:


plt.figure(figsize= (30,30))
dc = sns.heatmap(df_train.corr() , annot = True , annot_kws={'size': 18})
dc.set_xticklabels(dc.get_xmajorticklabels(), fontsize = 18)
dc.set_yticklabels(dc.get_ymajorticklabels(), fontsize = 18)
dc


# Thus, using Dummies Feature , No Features are left correlated with each other

# # Training our Model

# Regarding our Model , we are interested in the Following 3 Things :
#     
# a) **Coefficient and its p-values** : If the p-value of the coefficient for any Feature is less than 0.05 , it means it is Statistically Significant.
# 
# b) **R^2 Values** : As the columns with p-values greater than 0.05 will get removed, our R^2 value will get better.
#     
# c) **F Statistics should have a very low p-value** : This means that our entire model is Statistically Significant.

# In[45]:


x_train = df_train
y_train = df_train.pop('Chance of Admit')

print(x_train.shape)
print(y_train.shape)


# In[46]:


x_train.head()


# In[47]:


# Keep X-Train Data for building the Model
x_train_1 = x_train.copy(deep = True)

# Adding the Constant Column
x_train_1 = sm.add_constant(x_train_1)

print(x_train_1.shape)
LR_1 = sm.OLS(y_train , x_train_1).fit()
LR_1.params


# In[48]:


LR_1.summary()


# From above , we can see that 
# - R score : 0.735
# - Based on the p-values, Features that are Eliminated are - University Rating_3 (0.826) , SOP_2 (0.847) , LOR_3.5 (0.753).

# In[49]:


x_train_2 = x_train[['GRE Score' , 'Research' , 'University Rating_2' , 'University Rating_4' , 'University Rating_5' , 'SOP_1.5' , 'SOP_2.5' , 'SOP_3.0' , 'SOP_3.5' , 'SOP_4.0' , 'SOP_4.5' , 'SOP_5.0' , 'LOR_1.5' , 'LOR_2.0' , 'LOR_2.5' , 'LOR_3.0' , 'LOR_4.0' , 'LOR_4.5' , 'LOR_5.0']]

x_train_2 = sm.add_constant(x_train_2)
LR_2 = sm.OLS(y_train , x_train_2).fit()

LR_2.summary()


# We can see that 
# 
# - After removal of 3 features , R Score hasn't changed.
# - Also, based on the p-values, we can see that Following Features should be removed : University Rating_2 (0.865) , SOP_1.5 (0.356) , LOR_4 (0.757).

# In[50]:


x_train_3 = x_train[['GRE Score' , 'Research' , 'University Rating_4' , 'University Rating_5' , 'SOP_2.5' , 'SOP_3.0' , 'SOP_3.5' , 'SOP_4.0' , 'SOP_4.5' , 'SOP_5.0' , 'LOR_1.5' , 'LOR_2.0' , 'LOR_2.5' , 'LOR_3.0' , 'LOR_4.5' , 'LOR_5.0']]

x_train_3 = sm.add_constant(x_train_3)
LR_3 = sm.OLS(y_train , x_train_3).fit()

LR_3.summary()


# We can see that 
# 
# - After removal of 3 features , R Score hasn't changed much 
# - Also, based on the p-values, we can see that Following Features should be removed : University Rating_4 (0.314) , SOP_3.0 (0.066) , LOR_2.5 (0.067).

# In[51]:


x_train_4 = x_train[['GRE Score' , 'Research' , 'University Rating_5' , 'SOP_2.5' , 'SOP_3.5' , 'SOP_4.0' , 'SOP_4.5' , 'SOP_5.0' , 'LOR_1.5' , 'LOR_2.0' , 'LOR_3.0' , 'LOR_4.5' , 'LOR_5.0']]

x_train_4 = sm.add_constant(x_train_4)
LR_4 = sm.OLS(y_train , x_train_4).fit()

LR_4.summary()


# We can see that
# 
# - After removal of 3 features , R Score hasn't changed much
# - Also, based on the p-values, we can see that Following Features should be removed : LOR_3 (0.088) , SOP_2.5 (0.041) , LOR_1.5 (0.014).

# In[52]:


x_train_5 = x_train[['GRE Score' , 'Research' , 'University Rating_5' , 'SOP_3.5' , 'SOP_4.0' , 'SOP_4.5' , 'SOP_5.0' , 'LOR_2.0' , 'LOR_4.5' , 'LOR_5.0']]

x_train_5 = sm.add_constant(x_train_5)
LR_5 = sm.OLS(y_train , x_train_5).fit()

LR_5.summary()


# We can see that
# 
# - After removal of 3 features , R Score hasn't changed much.
# - Also, based on the p-values, we can see that Following Features should be removed : SOP_4 (0.004) , SOP_3.5 (0.012) , LOR_2.0 (0.012).

# In[53]:


x_train_6 = x_train[['GRE Score' , 'Research' , 'University Rating_5' , 'SOP_4.5' , 'SOP_5.0' , 'LOR_4.5' , 'LOR_5.0']]

x_train_6 = sm.add_constant(x_train_6)
LR_6 = sm.OLS(y_train , x_train_6).fit()

LR_6.summary()


# We can see that
# 
# - After removal of 3 features , R Score hasn't changed much.
# - Also, based on the p-values, we can see that Following Features should be removed : SOP_4.5 (0.038) , SOP_5.0 (0.040).

# In[54]:


x_train_7 = x_train[['GRE Score' , 'Research' , 'University Rating_5' , 'LOR_4.5' , 'LOR_5.0']]

x_train_7 = sm.add_constant(x_train_7)
LR_7 = sm.OLS(y_train , x_train_7).fit()

LR_7.summary()


# In[55]:


parameters = round(LR_7.params,4)
parameters


# In[56]:


plt.figure(figsize= (5,5))
d_new = sns.heatmap(x_train_7.corr() , annot = True , annot_kws={'size': 10})
d_new


# # Observations : 

# - After removal of several Features, R - Score values haven't changed much.
# - Now that the p-values for our Columns have reached 0, we can say that our Best-Fit model is attained.
# - Thus , the Model coefficients attained are : 
# 
#     const                   :  -0.3133
#     
#     GRE Score               :    0.5970
#     
#     Research                :    0.2777
#     
#     University Rating_5     :    0.4363
#     
#     LOR_4.5                 :    0.3743
#     
#     LOR_5.0                 :    0.4350
#     

# # Assumptions of Linear Regression :

# - **Linearity**: The relationship between X and the mean of Y is linear.
# - **Homoscedasticity**: The variance of residual is the same for any value of X.
# - **Independence**: Observations are independent of each other.
# - **Normality**: For any fixed value of X, Y is normally distributed.
# - **No Autocorrelation** : Error Data shouldn't form any Patterns in its Plots.

# # 1. Multicollinearity Check - Using VIF Score :-

# In[57]:


vif = pd.DataFrame()
vif["VIF Values"] = [variance_inflation_factor(x_train_7.values , i) for i in range(x_train_7.shape[1])]
vif["Features"] = x_train_7.columns
vif


# Since all Features are below 5, means No Multicollinearity.

# In[58]:


df_pred = df_test.copy(deep = True)
df_test.shape


# In[59]:


y_test = df_test.pop('Chance of Admit')
x_test = df_test
x_test = sm.add_constant(x_test)


# In[60]:


x_train_7.columns


# In[61]:


x_test_new = x_test[x_train_7.columns]

y_pred = LR_7.predict(x_test_new)


# In[62]:


residual_test = y_test - y_pred
sns.boxplot(x = residual_test)


# # 2. Mean of Residuals :-

# In[63]:


residual = y_test - y_pred
sns.distplot(residual)
plt.show()


# In[64]:


abs(residual.mean())


# From this, It can be seen that our Model's Mean Residual is close to Zero , means it's a Good Model.

# # 3. Linearity of Variables :-

# In[65]:


plt.scatter(y_test , y_pred)
plt.title(" Test Output vs Prediction Output" , fontsize = 20)
plt.xlabel("Test Output (y_test)", fontsize = 18)
plt.ylabel("Prediction Output (y_pred)", fontsize = 18)
plt.show()


# # 4. Homoscedasticity Testing :

# Homoscedasticity means that the residuals have equal or almost equal variance across the regression line. By plotting the error terms with predicted terms we can check that there should not be any pattern in the error terms.
# 
# There are 2 ways -
# 1. By checking the Plots Graphically
# 2. By applying Goldfeld Quandt Test (and aiming for Null Hypothesis)

# # a) Graphical Approach - 

# In[66]:


fig = plt.figure(figsize=(20,20))

# Using statsmodels.graphics.regressionplots.plot_regress_exog to Plot Regression Results against one Regressor
fig = sm.graphics.plot_regress_exog(LR_7 , "GRE Score" , fig = fig)
fig.tight_layout(pad=1.0)
plt.show()


# It can be seen that the points are spread Randomly, and Points of Residual are scattered around Zero. Hence, There is no Heteroscedasticity.

# # b) Goldfeld Quandt Test - 

# Checking Heteroscedasticity : Using Goldfeld Quandt , we test for heteroscedasticity.
# 
# - Null Hypothesis: Error terms are homoscedastic
# - Alternative Hypothesis: Error terms are heteroscedastic.

# In[67]:


import statsmodels.stats.api as sms
from statsmodels.compat import lzip

name = ['F Statistic' , 'p-value']
test = sms.het_goldfeldquandt(y_train , x_train_7)
lzip(name , test)


# # Observations : 

# Since the p-value from Hypothesis Testing is more than the Significance Value (0.05), this means that we fail to reject the Null Hypothesis.
# Thus , using both the Graphical Approach and Goldfeld Quandt Test Approach , we see that Heteroscedasticity isn't present in our Model.

# # 5. Normality of Residues : 

# In[68]:


sm.qqplot(residual , line = 's')
plt.show()


# In[69]:


h = sns.distplot(residual , kde = True)
h = plt.title('Residual Normality Check')


# From above, it can se seen that the Data is aligned closely to the Dotted line. Hence, Residuals are Normally Distributed.
# The Distplot also shows that the Residual Terms are nearly Normally Distributed.

# # 6. No Autocorrelation of Residuals :

# When the residuals are autocorrelated, it means that the current value is dependent of the previous (historic) values and that there is a definite unexplained pattern in the Y variable that shows up in the error terms.
# There should not be autocorrelation in the data so the error terms should not form any pattern.

# In[70]:


plt.figure(figsize = (5,5))
auto_graph = sns.lineplot(y_pred , residual , marker = 'o' , color = 'red')
plt.xlabel('Prediction Values')
plt.ylabel('Residuals')
plt.ylim(-1,1)
plt.xlim(0,2.2)
p = sns.lineplot([0,2.2],[0,0],color = 'blue')
p = plt.title('Residuals vs Fitted values plot for AutoCorrelation Check')


# This means that No Auto-correlation exits.
# 
# Since all of the conditions are met, this means that Our Model is good to consider for Further Predictions.

# # Model Performance Evaluation : 

# In[71]:


r2 = r2_score(y_test , y_pred)
adj_r2 = LR_7.rsquared_adj
mae = mean_absolute_error(y_test , y_pred)
mse = mean_squared_error(y_test , y_pred)
rmse = np.sqrt(mean_squared_error(y_test , y_pred))

print('R Square Value :', round(r2,2))
print('Adjusted R Square Value :', round(adj_r2,2))
print('Mean Absolute Error Value :', round(mae,2))
print('Mean Square Error Value :', round(mse,2))
print('Root Mean Square Error Value :', round(rmse,2))


# # Observations : 

# - As r2 metric value is 0.76, this means that our model is a Good Model, since r2 value is neither too good nor too low.
# - Since MAE error (0.37) is on the lower side , this means that Our Model is Good but not the Perfect Model. 
# - Since difference between MAE error (0.37) and RMSE error (0.48) is on the Lower Side , this shows Less Outliers.

# # Test and Train Performances : 

# In[72]:


# Train Performance

y_train_pred = LR_7.predict(x_train_7)
res = y_train - y_train_pred

sns.distplot(res)
plt.title("Train Data Residual Performance")
plt.show()


# In[73]:


# Test Performance 

df_pred['predicted'] = y_pred

sns.lmplot(x = 'Chance of Admit' , y = 'predicted' , data = df_pred)
plt.xlabel('Actual Values' , fontsize = 16)
plt.ylabel('Predicted Values' , fontsize = 16)
plt.title('Actual Values vs Predicted Values' , fontsize = 20)
plt.show()


# In[74]:


sns.kdeplot(data = df_pred , x = "Chance of Admit" , color = 'b' , multiple = 'stack' , label = 'Actual Values')
sns.kdeplot(data = df_pred , x = 'predicted' , color = 'g' , multiple = 'stack' , label = 'Predicted Values')
plt.show()


# # Observations :

# These above plots show that our Model can be considered for our Purpose.

# # Lasso Regression or L1 - Regularization : 

# Lasso regression performs L1 regularization, which adds a penalty equal to the absolute value of the magnitude of coefficients. This type of regularization can result in sparse models with few coefficients; Some coefficients can become zero and eliminated from the model. Larger penalties result in coefficient values closer to zero, which is the ideal for producing simpler models.

# In[75]:


from sklearn.linear_model import Lasso


# In[76]:


regressor_lasso = Lasso(alpha = 10.0)
regressor_lasso.fit(x_train_7 , y_train)


# In[77]:


print(regressor_lasso.intercept_)
print(regressor_lasso.coef_)
print(x_train_7.columns)


# In[78]:


# Now we can find out which Features are Important and which aren't Important Features

importance = regressor_lasso.coef_
features = x_train_7.columns

print(np.array(features)[importance != 0])
print(np.array(features)[importance == 0])


# In[79]:


# Residual Analysis on Training Data for L1 Regularization

y_train_pred_lasso = regressor_lasso.predict(x_train_7)
residual_lasso = y_train - y_train_pred_lasso
sns.displot(residual_lasso, kind='kde')


# In[80]:


plt.scatter(y_train, residual_lasso)
plt.show()


# In[81]:


x_test = x_test[['const' , 'GRE Score' , 'Research', 'University Rating_5', 'LOR_4.5', 'LOR_5.0']]
x_test


# In[82]:


# Prediction for L1

y_test_pred_lasso = regressor_lasso.predict(x_test)
temp_df_L1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred_lasso})
temp_df_L1.head()


# In[83]:


from sklearn import metrics
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_test_pred_lasso))

print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_test_pred_lasso))

print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_lasso)))


# In[84]:


residual_test_lasso = y_test - y_test_pred_lasso
sns.boxplot(x = residual_test_lasso)


# # Ridge Regression or L2 - Regularization : 

# L2 regularization adds an L2 penalty, which equals the square of the magnitude of coefficients. All coefficients are shrunk by the same factor (so none are eliminated). Unlike L1 regularization, L2 will not result in sparse models.

# In[85]:


from sklearn.linear_model import Ridge


# In[86]:


regressor_ridge = Ridge(alpha = 10.0)
regressor_ridge.fit(x_train_7 , y_train)


# In[87]:


print(regressor_ridge.intercept_)
print(regressor_ridge.coef_)
print(x_train_7.columns)


# In[88]:


# Now we can find out which Features are Important and which aren't Important Features

importance = regressor_ridge.coef_
features = x_train_7.columns

print(np.array(features)[importance != 0])
print(np.array(features)[importance == 0])


# In[89]:


# Residual Analysis on Training Data for L2 Regularization

y_train_pred_ridge = regressor_ridge.predict(x_train_7)
residual_ridge = y_train - y_train_pred_ridge
sns.displot(residual_ridge, kind='kde')


# In[90]:


plt.scatter(y_train, residual_ridge)
plt.show()


# In[91]:


# Prediction for L2

y_test_pred_ridge = regressor_ridge.predict(x_test)
temp_df_L2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred_ridge})
temp_df_L2.head()


# In[92]:


from sklearn import metrics
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_test_pred_ridge))

print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_test_pred_ridge))

print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_ridge)))


# In[93]:


residual_test_ridge = y_test - y_test_pred_ridge
sns.boxplot(x = residual_test_ridge)


# # Comparison for L1 , L2 Regularizations : 

# In[94]:


R_L1 = r2_score(y_test , y_test_pred_lasso)
R_L2 = r2_score(y_test , y_test_pred_ridge)
print(R_L1)
print(R_L2)


# In[95]:


print('Mean Absolute Error (L1 Regularization) : ', metrics.mean_absolute_error(y_test, y_test_pred_lasso))

print('Mean Squared Error (L1 Regularization) : ', metrics.mean_squared_error(y_test, y_test_pred_lasso))

print('Root Mean Squared Error (L1 Regularization) : ', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_lasso)))


# In[96]:


print('Mean Absolute Error (L2 Regularization) : ', metrics.mean_absolute_error(y_test, y_test_pred_ridge))

print('Mean Squared Error (L2 Regularization) : ', metrics.mean_squared_error(y_test, y_test_pred_ridge))

print('Root Mean Squared Error (L2 Regularization) : ', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_ridge)))


# In[97]:


res_df = pd.DataFrame({'Model Residuals': residual_test, 'L1 Model Residuals': residual_test_lasso , 'L2 Model Residuals': residual_test_ridge})
print(res_df)
print("Mean Residual without Regularization :", res_df['Model Residuals'].mean())
print("L1 Regularization Mean Residual :", res_df['L1 Model Residuals'].mean())
print("L2 Regularization Mean Residual :", res_df['L2 Model Residuals'].mean())
sns.boxplot(data = res_df, orient='h')


# # Observations : 

# - Since the Error Values for L2 Regularization seems lesser than those of L1 Regularization, and the r2 metric for L1 Regularization is extremely bad, thus L2 Regularization should be considered here.
# - As per L1 Regularization , all the columns seem to be irrelevant , but in L2 Regularization, only the column 'const' seem unimportant.  

# # Insights :

# - Considering the University Rating aspect , Maximum Students were enrolled in the Universities having Ratings of 3.
# - Considering the SOP aspect , Maximum Students had a SOP Ratings of 3 and 4.
# - Considering the LOR aspect , Maximum Students had a LOR Rating of 3.
# - Considering the Research aspect , More number of Students had a Prior Research Experience.
# - Numerical Columns like GRE Scores , TOEFL Scores and CGPA are following a Linear Relationship with respect to Chances of Admit.
# - As per the Pairplot for Research , Students considering themselves towards Research Domain have higher Chances of getting Admission, in comparison to those who don't consider Research into consideration.
# - As per the Pairplot for University Ranking , Students require better GRE , TOEFL and CGPA Scores to get admission into better Universities. Considering the Data , Maximum number of Students got admissions in Universities with Ratings 3. And minimum Number of Students got admission in Universities with Ratings 1.
# - Column 'Chance of Admit' has a Positive Linear Relation with all of the Numerical and Categorical Features.
# 
# - Chance of Admit vs Research : 
# 
#   Having Research has more Median Value compared to not having Research. Also, Research Students have higher chances than Non-Researchers. Outliers are present only for Research Aspect.
#   
#   
# - Chance of Admit vs University Rating : 
# 
#   As the University Rating increases, the chances of getting admission also Increases, since Rating '5' has the highest Median value followed by 4 , 3 , 2 and 1. Outliers present for most of the Categories.
#   
#   
# - Chance of Admit vs SOP : 
# 
#   As the SOP Rating increases, the chances of getting admission also Increases, since Rating '5' has the highest Median value followed by others. Outliers present for most of the Categories.
#   
#   
# - Chance of Admit vs LOR : 
# 
#   As the LOR Rating increases, the chances of getting admission also Increases, since Rating '5' has the highest Median value followed by others. Outliers present for most of the Categories.
#   
# - Thus , the Model coefficients attained are : 
# 
#     const                   :  -0.3133
#     
#     GRE Score               :    0.5970
#     
#     Research                :    0.2777
#     
#     University Rating_5     :    0.4363
#     
#     LOR_4.5                 :    0.3743
#     
#     LOR_5.0                 :    0.4350 
#     
#     Means that GRE Score has the Highest Impact on the chances of any Student getting Admission into any University.
#   
#   
# - Our Final Model is passing all the Assumptions for Linear Regression correctly, thus the Metric Values for our Model are : 
# 
#   R Square Value : 0.76 , 
#   Adjusted R Square Value : 0.69 , 
#   Mean Absolute Error Value : 0.37 , 
#   Mean Square Error Value : 0.23 , and 
#   Root Mean Square Error Value : 0.48. 
#   
# - Metric Values for L1 Regularization are : 
# 
#   R Square Value : -0.0023 , 
#   Mean Absolute Error Value : 0.78 , 
#   Mean Square Error Value : 0.96 , and 
#   Root Mean Square Error Value : 0.98. 
#   
#   
# - Metric Values for L2 Regularization are : 
# 
#   R Square Value : 0.76 , 
#   Mean Absolute Error Value : 0.36 , 
#   Mean Square Error Value : 0.22 , and 
#   Root Mean Square Error Value : 0.47. 
#   
#   Thus , L2 Regularization should be preferred over L1 Regularization for our Model.

# # Recommendations : 

# - Main features which influence the chance of Admit are: GRE Score , Research , TOEFL Score , CGPA , LOR greater or equal to than 4.5.
#   For any Applicant , GRE Score should be the Primary Target.
# - A Higher University rating will increases the chance of admission
# - A Higher Value of LOR and SOP will also increase the chance of admission for the student.
