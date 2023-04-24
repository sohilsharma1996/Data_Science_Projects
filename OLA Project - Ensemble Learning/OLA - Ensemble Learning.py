#!/usr/bin/env python
# coding: utf-8

# Recruiting and retaining drivers is seen by industry watchers as a tough battle for Ola. Churn among drivers is high and it’s very easy for drivers to stop working for the service on the fly or jump to Uber depending on the rates.
# 
# As the companies get bigger, the high churn could become a bigger problem. To find new drivers, Ola is casting a wide net, including people who don’t have cars for jobs. But this acquisition is really costly. Losing drivers frequently impacts the morale of the organization and acquiring new drivers is more expensive than retaining existing ones.

# # Problem Statement : 

# You are working as a data scientist with the Analytics Department of Ola, focused on driver team attrition. You are provided with the monthly information for a segment of drivers for 2019 and 2020 and tasked to predict whether a driver will be leaving the company or not based on their attributes like : 
# 
# - Demographics (city, age, gender etc.)
# - Tenure information (joining date, Last Date)
# - Historical data regarding the performance of the driver (Quarterly rating, Monthly business acquired, grade, Income)

# # Data Description : 

# - MMMM-YY : Reporting Date (Monthly)
# - Driver_ID : Unique id for drivers
# - Age : Age of the driver
# - Gender : Gender of the driver – Male : 0, Female: 1
# - City : City Code of the driver
# - Education_Level : Education level – 0 for 10+ ,1 for 12+ ,2 for graduate
# - Income : Monthly average Income of the driver
# - Date Of Joining : Joining date for the driver
# - LastWorkingDate : Last date of working for the driver
# - Joining Designation : Designation of the driver at the time of joining
# - Grade : Grade of the driver at the time of reporting
# - Total Business Value : The total business value acquired by the driver in a month (negative business indicates cancellation/refund or car EMI adjustments)
# - Quarterly Rating : Quarterly rating of the driver: 1,2,3,4,5 (higher is better)

# # Import the Libraries : 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns' , None)
pd.set_option('display.max_rows' , None)


# In[2]:


import matplotlib as mpl
import plotly.graph_objs as go
import matplotlib.patches as mpatches
import plotly.express as px
from plotly import tools
from plotly.subplots import make_subplots
from plotly.offline import iplot


# In[3]:


df = pd.read_csv("ola_driver_scaler.csv")
df.head()


# In[4]:


df.ndim


# In[5]:


df.shape


# In[6]:


df.size


# In[7]:


df = df.drop(['Unnamed: 0'], axis=1)
df.head()


# In[8]:


df.info()


# In[9]:


sns.heatmap(df.isnull() ,cbar=False)


# In[10]:


sns.heatmap(df.isna() ,cbar=False)


# In[11]:


df.isnull().sum()/len(df.index)*100


# In[12]:


df.columns


# In[13]:


df.index


# In[14]:


df.dtypes


# In[15]:


df['Age'].unique()


# In[16]:


categorical_columns = ['Age' , 'Gender' , 'Education_Level' , 'Income' , 'Joining Designation' , 'Grade']


# In[17]:


# df['Attrition'] = df['LastWorkingDate'].fillna(0)
# df['Attrition'] = df['LastWorkingDate'].notnull()
# df['Attrition'] = df['Attrition'].map({False:0, True: 1})


# In[18]:


sns.heatmap(df.isnull() ,cbar=False)


# In[19]:


df.head()


# In[20]:


df['MMM-YY'] = pd.to_datetime(df['MMM-YY'])
df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'])
df['LastWorkingDate'] = pd.to_datetime(df['LastWorkingDate'])


# In[21]:


df.info()


# In[22]:


categorical_features = list(df.select_dtypes('object').columns)


# In[23]:


for i in categorical_features:
    print('Unique Values in {0} are {1}'.format(i,df[i].unique()))


# In[24]:


df["Driver_ID"].value_counts().head(20)


# In[25]:


import missingno as mn
mn.matrix(df,color=(0.40,0.20,1))


# In[26]:


#PERCENTAGE OF THE MISSING VALUES - DATAFRAME..... 
def missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    Percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, Percentage], axis=1, keys=['Total', 'Percentage'])


# In[27]:


missing_data(df).style.set_properties(**{"background-color": "#AB63FA","color": "white", "border-color": "white","font-size":"11.5pt",'width': 200})


# In[28]:


df[df.duplicated()]


# In[29]:


round(df.describe(exclude = ['object',"datetime64[ns]"]), 2)


# In[30]:


round(df.describe(exclude = ["datetime64[ns]",'float', 'int64']),2)


# In[31]:


round(df.describe(exclude = ["object",'float', 'int64']),2)


# In[32]:


df.head()


# In[33]:


df_1 = pd.DataFrame()


# In[34]:


df_1['Driver_ID'] = df['Driver_ID'].unique()

#Setting age of the employee which takes the maximum age of the employee
df_1['Age'] = list(df.groupby('Driver_ID',axis=0).max('MMM-YY')['Age'])

#Setting gender of the employee 
df_1['Gender'] = list(df.groupby('Driver_ID').agg({'Gender':'first'})['Gender'])

#Setting the city of the employee 
df_1['City'] = list(df.groupby('Driver_ID').agg({'City':'first'})['City'])

#Setting the education of the employee 
df_1['Education'] = list(df.groupby('Driver_ID').agg({'Education_Level':'last'})['Education_Level'])

#Setting the salary of the employee for one month
df_1['Income'] = list(df.groupby('Driver_ID').agg({'Income':'last'})['Income'])

#Setting the joining designtion of the employee 
df_1['Joining_Designation'] = list(df.groupby('Driver_ID').agg({'Joining Designation':'last'})['Joining Designation'])

#Setting the designtion of the employee at the time of reporting            
df_1['Designation'] = list(df.groupby('Driver_ID').agg({'Grade':'last'})['Grade'])


# In[35]:


z = df.groupby('Driver_ID',axis=0).sum('Total Business Value')['Total Business Value']
z.head(5)


# In[36]:


y = df.groupby('Driver_ID',axis=0).sum({'Quarterly Rating':'last'})['Quarterly Rating']
y.head(5)


# In[37]:


df_1['Total_Business_Value'] = list(z)


# In[38]:


df_1['Last_Quarterly_Rating'] = list(y)


# In[39]:


#Creating a column which tells if the quarterly rating has increased for that employee 
#for those whose quarterly rating has increased we assign the value 1

#Quarterly rating at the beginning
qrf = df.groupby('Driver_ID').agg({'Quarterly Rating':'first'})

#Quarterly rating at the end
qrl = df.groupby('Driver_ID').agg({'Quarterly Rating':'last'})
#The dataset which has the employee ids and a bollean value which tells if the rating has increased
qr = (qrl['Quarterly Rating'] > qrf['Quarterly Rating']).reset_index()

#the employee ids whose rating has increased
driverid = qr[qr['Quarterly Rating']==True]['Driver_ID']

qri = []
for i in df_1['Driver_ID']:
    if i in driverid:
        qri.append(1)
    else:
        qri.append(0)

df_1['Quarterly_Rating_Increased'] = qri


# In[40]:


#Creating a column called target which tells if the person has left the company
#persons who have a last working date will have the value 1

#The dataset which has the employee ids and specifies if last working date is null
lwr = (df.groupby('Driver_ID').agg({'LastWorkingDate':'last'})['LastWorkingDate'].isna()).reset_index()

#The employee ids who do not have last working date
driverid = list(lwr[lwr['LastWorkingDate']==True]['Driver_ID'])

Attrition = []
for i in df_1['Driver_ID']:
    if i in driverid:
        Attrition.append(0)
    elif i not in driverid:
        Attrition.append(1)
        
df_1['Attrition'] = Attrition


# In[41]:


df_1.head(5)


# In[42]:


#Created a new column called "promotion" because there is a chances of employee leaving company due to not getting promoted...may be!
df_1['Promotion'] = np.where(df_1['Designation'] > df_1['Joining_Designation'] , 1 , 0)


# In[43]:


df_1.head()


# In[44]:


df['LastWorkingDate'].fillna(0,inplace = True)
df['LastWorkingDate'] = df['LastWorkingDate'].apply(lambda x:0 if x==0 else 1)


# In[45]:


old_df = df[['Driver_ID','MMM-YY','City','Dateofjoining']]


# In[46]:


new_df = df[['Education_Level','Age','Gender','Income','Joining Designation','Grade','Total Business Value','Quarterly Rating','LastWorkingDate']]


# In[47]:


from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
im = KNNImputer(n_neighbors=5)
new_df = pd.DataFrame(im.fit_transform(new_df),columns=new_df.columns)


# In[48]:


frames= [old_df,new_df]
df = pd.concat(frames,axis=1)


# # Univariate Analysis on Numerical Features : 

# In[49]:


# list of numerical variables............
numerical_features = [feature for feature in df_1.columns if ((df_1[feature].dtypes != 'O') & (df_1[feature].dtypes != 'datetime64[ns]'))]

print('Number of numerical variables: ', len(numerical_features))
print('\n')
print('Numberical Variables Column: ',numerical_features)
print('\n')
# visualise the numerical variables........
df_1[numerical_features].head().style.set_properties(**{"background-color": "#AB63FA","color": "white", "border-color": "white","font-size":"11.5pt",'width': 200})


# In[50]:


numerical_features


# In[51]:


## Lets analyse the numerical values by creating histograms to understand the distribution


# In[52]:


data = df_1.copy()
fig = px.histogram(df_1, x = 'Driver_ID' , color_discrete_sequence = ['#AB63FA'] , marginal = "box" , title="Distribution Of The ID")
fig.update_layout(bargap = 0.2 , template = 'plotly_white')


# In[53]:


fig = px.histogram(df_1, x = 'Age' , color_discrete_sequence = ['#AB63FA'] , marginal = "box" , title="Distribution Of The Age")
fig.update_layout(bargap = 0.2 , template = 'plotly_white')


# In[54]:


fig = px.histogram(df_1, x = 'Gender' , color_discrete_sequence = ['#AB63FA'] , marginal = "box" , title="Distribution Of The Gender")
fig.update_layout(bargap = 0.2 , template = 'plotly_white')


# In[55]:


fig = px.histogram(df_1, x = 'Education' , color_discrete_sequence = ['#AB63FA'] , marginal = "box" , title="Distribution Of The Education")
fig.update_layout(bargap = 0.2 , template = 'plotly_white')


# In[56]:


fig = px.histogram(df_1, x = 'Income' , color_discrete_sequence = ['#AB63FA'] , marginal = "box" , title="Distribution Of The Income")
fig.update_layout(bargap = 0.2 , template = 'plotly_white')


# In[57]:


fig = px.histogram(df_1, x = 'Joining_Designation' , color_discrete_sequence = ['#AB63FA'] , marginal = "box" , title="Distribution Of The Joining Designation")
fig.update_layout(bargap = 0.2 , template = 'plotly_white')


# In[58]:


fig = px.histogram(df_1, x = 'Designation' , color_discrete_sequence = ['#AB63FA'] , marginal = "box" , title="Distribution Of The Designation")
fig.update_layout(bargap = 0.2 , template = 'plotly_white')


# In[59]:


fig = px.histogram(df_1, x = 'Total_Business_Value' , color_discrete_sequence = ['#AB63FA'] , marginal = "box" , title="Distribution Of The Total Business Value")
fig.update_layout(bargap = 0.2 , template = 'plotly_white')


# In[60]:


fig = px.histogram(df_1, x = 'Last_Quarterly_Rating' , color_discrete_sequence = ['#AB63FA'] , marginal = "box" , title="Distribution Of The Last Quarterly Rating")
fig.update_layout(bargap = 0.2 , template = 'plotly_white')


# In[61]:


fig = px.histogram(df_1, x = 'Quarterly_Rating_Increased' , color_discrete_sequence = ['#AB63FA'] , marginal = "box" , title="Distribution Of The Quarterly Rating Increased")
fig.update_layout(bargap = 0.2 , template = 'plotly_white')


# In[62]:


fig = px.histogram(df_1, x = 'Attrition',color_discrete_sequence=['#AB63FA'],marginal="box",title="Distribution Of The Attrition")
fig.update_layout(bargap=0.2,template='plotly_white')


# In[63]:


fig = px.histogram(df_1, x = 'Promotion',color_discrete_sequence=['#AB63FA'],marginal="box",title="Distribution Of The Promotion")
fig.update_layout(bargap=0.2,template='plotly_white')


# In[64]:


print('Maximum age of the students:',df['Age'].max())
print('Manimum age of the students:',df['Age'].min())
print('Average age of the students:',df['Age'].mean())


# In[65]:


df_1.loc[(df['Age']>18)&(df['Age']<=25),'Emp_Age'] = 0
df_1.loc[(df['Age']>25)&(df['Age']<=35),'Emp_Age'] = 1
df_1.loc[(df['Age']>35)&(df['Age']<=45),'Emp_Age'] = 2
df_1.loc[(df['Age']>45)&(df['Age']<=55),'Emp_Age'] = 3
df_1.loc[df['Age']>55,'Emp_Age'] = 4

# converting 'Weight' from float to int
df_1['Emp_Age'] = df_1['Emp_Age'].astype(int)


# In[66]:


df_1['Emp_Age'].dtypes


# In[67]:


fig = px.histogram(df_1, x = 'Emp_Age',color_discrete_sequence=['#AB63FA'],title="Distribution Of The Emp_Age",marginal="box")
fig.update_layout(bargap=0.2,template='plotly_white')


# # Univariate Analysis on Categorical Features : 

# In[68]:


# list of Categorical variables............
cat_features = [feature for feature in df_1.columns if ((df_1[feature].dtypes == 'O') & (df_1[feature].dtypes != 'datetime64[ns]'))]

print('Number of categorical variables: ', len(cat_features))
print('\n')
print('Categorical variables columns are: ',cat_features)
print('\n')
# visualise the numerical variables........
df_1[cat_features].head().style.set_properties(**{"background-color": "#AB63FA","color": "white", "border-color": "white","font-size":"11.5pt",'width': 200})


# In[69]:


fig = px.histogram(df_1 , x = 'City' , color_discrete_sequence = ['#AB63FA'] , title="Distribution Of The City" , marginal="box")
fig.update_layout( bargap=0.2 , template='plotly_white' )


# # Observations : 

# - Out of 2381 employees, 1404 employees are of the Male gender and 977 are females.
# - Out of 2381 employees, 152 employees are from city C20 and 101 from city C15.
# - Out of 2381 employees, 802 employees have their education as Masters and 795 have completed their Bachelors.
# - Out of 2381 employees, 1026 joined with the designation as 1, 815 employees joined with the designation 2.
# - Out of 2381 employees, 855 employees had their designation as 2 at the time of reporting.
# - Out of 2381 employees, 1744 employees had their last quarterly rating as 1.
# - Out of 2381 employees, the quarterly rating has not increased for 2076 employees.
# - Around 6.4% employees are from city C20 and 4.2% from city C15.
# - The proportion of the employees who have completed their Masters and Bachelors is approximately same.
# - Around 43% of the employees joined with the designation 1.
# - At the time of reporting, 34% of the employees had their designation as 2.
# - Around 73% of the employees had their last quarterly rating as 1.
# - The quarterly rating has not increased for around 87% employees.

# # Bivariate Analysis : 

# # a) Driver Age vs Attrition : 

# In[70]:


pd.crosstab(df_1.Emp_Age,df_1.Attrition,margins=True).style.set_properties(**{"background-color": "#AB63FA","color": "white", "border-color": "white","font-size":"11.5pt",'width': 200})


# In[71]:


fig = px.histogram(df_1, x="Emp_Age", color ="Attrition", pattern_shape="Attrition",
                   template='plotly_white', barmode='group',color_discrete_sequence=['#AB63FA'],title= "How does the employee age impact attrition?")
fig.update_layout(bargap=0.2,template='plotly_white')


# Most of the Employees whose age is between 25 to 45 are leaving the organization.

# # b) Gender vs Attrition : 

# In[72]:


pd.crosstab(df_1.Gender,df_1.Attrition,margins=True).style.set_properties(**{"background-color": "#AB63FA","color": "white", "border-color": "white","font-size":"11.5pt",'width': 200})


# In[73]:


fig = px.histogram(df_1, x= "Gender", color ="Attrition", pattern_shape="Attrition",
                   template='plotly_white', barmode='group',color_discrete_sequence=['#AB63FA'],title= "How does the gender impact attrition?")
fig.update_layout(bargap=0.2,template='plotly_white')


# Most of the employees are male who are leaving the organization.

# # c) City vs Attrition : 

# In[74]:


pd.crosstab(df_1.City,df_1.Attrition,margins=True).style.set_properties(**{"background-color": "#AB63FA","color": "white", "border-color": "white","font-size":"11.5pt",'width': 200})


# In[75]:


fig = px.histogram(df_1, x="City", color ="Attrition", pattern_shape="Attrition",
                   template='plotly_white', barmode='group',color_discrete_sequence=['#AB63FA'],title= "How does the city impact attrition?")
fig.update_layout(bargap=0.2,template='plotly_white')


# Most of the employees who are fom "C20" are leaving the organization.

# # d) Education vs Attrition : 

# In[76]:


pd.crosstab(df_1.Education ,df_1.Attrition,margins=True).style.set_properties(**{"background-color": "#AB63FA","color": "white", "border-color": "white","font-size":"11.5pt",'width': 200})


# In[77]:


fig = px.histogram(df_1, x="Education", color ="Attrition", pattern_shape="Attrition",
                   template='plotly_white', barmode='group',color_discrete_sequence=['#AB63FA'],title= "How does the education level impact attrition?")
fig.update_layout(bargap=0.2,template='plotly_white')


# Looks like leaving organisation is not at all related to educational level.

# # e) Salary vs Attrition : 

# In[78]:


fig = px.histogram(df_1, x="Income", color ="Attrition", pattern_shape="Attrition",
                   template='plotly_white', barmode='group',color_discrete_sequence=['#AB63FA'],title="How does the income impact attrition?")
fig.update_layout(bargap=0.2,template='plotly_white')


# - Most of the employees whose salary is between 20k-80k are leaving the organization.
# - Obviosly if the employee is earning good salary that why would he/she leave the job.

# # f) Promotion vs Attrition : 

# In[79]:


pd.crosstab(df_1['Promotion'] ,df_1.Attrition,margins=True).style.set_properties(**{"background-color": "#AB63FA","color": "white", "border-color": "white","font-size":"11.5pt",'width': 200})


# In[80]:


fig = px.histogram(df_1, x="Promotion", color ="Attrition", pattern_shape="Attrition",
                   template='plotly_white', barmode='group',color_discrete_sequence=['#AB63FA'],title= "How does the Promotion impact attrition?")
fig.update_layout(bargap=0.2,template='plotly_white')


# - Most of the employees are not getting promoted may be that is the reason of leaving the organization.
# - Very few employees are getting acknowledged their work at the organization.

# # g) Total Business Value vs Attrition : 

# In[81]:


pd.crosstab(df_1['Total_Business_Value'] ,df_1.Attrition,margins=True).style.set_properties(**{"background-color": "#AB63FA","color": "white", "border-color": "white","font-size":"11.5pt",'width': 200})


# In[82]:


fig = px.histogram(df_1, x="Total_Business_Value", color ="Attrition", pattern_shape="Attrition",
                   template='plotly_white', barmode='group',color_discrete_sequence=['#AB63FA'],title= "How does the Total_Business_Value impact attrition?")
fig.update_layout(bargap=0.2,template='plotly_white')


# Most of the employees whose total business values is 5% are the highest no. of employees who are leaving the organization.

# # h) Quarterly Increased Rating vs Attrition : 

# In[83]:


pd.crosstab(df_1['Quarterly_Rating_Increased'] ,df_1.Attrition,margins=True).style.set_properties(**{"background-color": "#AB63FA","color": "white", "border-color": "white","font-size":"11.5pt",'width': 200})


# In[84]:


fig = px.histogram(df_1, x="Quarterly_Rating_Increased", color ="Attrition", pattern_shape="Attrition",
                   template='plotly_white', barmode='group',color_discrete_sequence=['#AB63FA'],title= "How does the Quarterly_Rating_Increased impact attrition?")
fig.update_layout(bargap=0.2,template='plotly_white')


# - Most of the employees are getting low rating may be could be the reason of leaving the organization.
# - Very few employees are getting acknowledged their work at the organization and getting good rating.

# In[85]:


fig, ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(df_1.corr(),cmap='YlGnBu',annot=True,annot_kws={'fontsize':10})
plt.show()


# In[86]:


import matplotlib
background_color = "#f6f6f6"

fig = plt.figure(figsize=(28,8), facecolor=background_color)
gs = fig.add_gridspec(1, 1)
ax0 = fig.add_subplot(gs[0, 0])
colors = ["#2f5586", "#f6f5f5","#2f5586"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

ax0.set_facecolor(background_color)
ax0.text(-1.1, 1.25, 'Correlation of Numerical Features with Target', fontsize=20, fontweight='bold')

chart_df = pd.DataFrame(df_1.corrwith(df_1['Attrition']))
chart_df.columns = ['corr']
sns.barplot(x=chart_df.index, y=chart_df['corr'], ax=ax0, color='#AB63FA', zorder=3, edgecolor='black', linewidth=3.5)
ax0.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
ax0.set_ylabel('')

for s in ["top","right", 'left']:
    ax0.spines[s].set_visible(False)
    
plt.show()


# After performing correlation we can say that Emp_Id , Quarterly_Rating_Increased , Salary_Increased are highly correlated.

# In[87]:


from sklearn.model_selection import train_test_split


# In[88]:


cols = ['Driver_ID', 'Age','Gender', 'Education' , 'Income',
       'Joining_Designation', 'Designation', 'Total_Business_Value',
       'Last_Quarterly_Rating' , 'Quarterly_Rating_Increased' , 'Promotion' , 'Emp_Age']
X = df_1[cols]
y = df_1['Attrition']

X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size=0.25 , shuffle=True )

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[89]:


from imblearn.over_sampling import SMOTE
from collections import Counter


# In[90]:


sm = SMOTE()
x_sm,y_sm = sm.fit_resample(X_train,y_train)
print('Resampled data has {}'.format(Counter(y_sm)))


# In[91]:


x_sm.head()


# In[92]:


y_sm.head()


# In[93]:


df_1 = pd.get_dummies(data = df_1 , columns = ['City'] , drop_first=True)
df_1.head().style.set_properties(**{"background-color": "#AB63FA","color": "white", "border-color": "white","font-size":"11.5pt",'width': 200})


# In[94]:


print("Value_count:",df_1['Attrition'].value_counts())
print("Shape:",df_1.shape)
fig = px.histogram(df_1, x = 'Attrition',color_discrete_sequence=['#AB63FA'],title="Imbalanced dataset")
fig.update_layout(bargap=0.2,template='plotly_white')


# In[95]:


#Using oversampling technique i am trying to balance the dataset...

from sklearn.utils import resample

#Separate majority and minority classes
df_majority = df_1[df_1.Attrition==1]
df_minority = df_1[df_1.Attrition==0]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, replace=True,n_samples=1616)      # sample with replacement

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

print(df_upsampled['Attrition'].value_counts())
X=df_upsampled['Attrition'].value_counts()


# In[96]:


#Rearrangement of the columns
df_1 = df_1[['Driver_ID','Emp_Age','City_C10', 'City_C11',
       'City_C12', 'City_C13', 'City_C14', 'City_C15', 'City_C16', 'City_C17',
       'City_C18', 'City_C19', 'City_C2', 'City_C20', 'City_C21', 'City_C22',
       'City_C23', 'City_C24', 'City_C25', 'City_C26', 'City_C27', 'City_C28',
       'City_C29', 'City_C3', 'City_C4', 'City_C5', 'City_C6', 'City_C7',
       'City_C8', 'City_C9','Income', 'Joining_Designation', 'Designation',
       'Total_Business_Value', 'Last_Quarterly_Rating',
       'Quarterly_Rating_Increased','Promotion', 'Attrition']]

df_1.head(5).style.set_properties(**{"background-color": "#AB63FA","color": "white", "border-color": "white","font-size":"11.5pt",'width': 200})


# # Standardization : 

# In[97]:


from sklearn.preprocessing import MinMaxScaler
#MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(x_sm)
df_scaled = scaler.transform(x_sm)
df_scaled


# In[98]:


df_2 = df_1.copy()
df_1.drop(columns = ["Total_Business_Value" , "Income" , "Driver_ID"] , inplace = True)


# # Model Creation using Bagging and Boosting : 

# In[99]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve ,KFold
from sklearn.metrics import roc_curve,accuracy_score,f1_score,roc_curve,confusion_matrix,roc_auc_score,plot_confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,ConfusionMatrixDisplay


# # Random Forest Model (Bagging):

# In[100]:


rf = RandomForestClassifier(max_depth= 7, n_estimators= 100)
rf.fit(X_train, y_train)


# In[101]:


rf_y_preds = rf.predict(X_test)

print(classification_report(y_test , rf_y_preds))


# In[102]:


print("The f1-score = ",f1_score(y_test,rf_y_preds))


# In[103]:


conf_matrix_rf = confusion_matrix(y_test,rf_y_preds)
conf_matrix_rf


# In[104]:


ConfusionMatrixDisplay(conf_matrix_rf).plot()


# In[105]:


np.diag(conf_matrix_rf).sum() / conf_matrix_rf.sum()*100


# # Gradient Boosting Model : 

# In[106]:


gb = GradientBoostingClassifier()

parameters = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.001, 0.1, 1, 10],
    'n_estimators': [100, 150, 180, 200]
}

grid_search = GridSearchCV(gb, parameters, cv = 5, n_jobs = -1, verbose = 1)
grid_search.fit(X_train, y_train)

print(f'Best parameters are : {grid_search.best_params_}')
print(f'The score is : {grid_search.best_score_}')


# Since Accuracy for Random Forest is more than that of Gradient Boosting, we can prefer Random Forest Technique.

# # ROC - AUC Curve : 

# In[107]:


from sklearn.metrics import RocCurveDisplay


# # For Random Forest : 

# In[108]:


RocCurveDisplay.from_estimator(rf, X_test, y_test)


# In[109]:


RocCurveDisplay.from_predictions(y_test, rf_y_preds)


# # Insights : 

# - Around 6.4% employees are from city C20 and 4.2% from city C15.
# - The proportion of the employees who have completed their Masters and Bachelors is approximately same.
# - Around 43% of the employees joined with the designation 1.
# - At the time of reporting, 34% of the employees had their designation as 2.
# - Around 73% of the employees had their last quarterly rating as 1.
# - The quarterly rating has not increased for around 87% employees.
# 
# - Most of the Employees whose age is between 25 to 45 are leaving the organization.
# - Most of the employees are male who are leaving the organization.
# - Most of the employees who are fom "C20" are leaving the organization.
# - Most of the employees whose salary is between 20k-80k are leaving the organization.
# - Most of the employees are not getting promoted may be that is the reason of leaving the organization.
# - Very few employees are getting acknowledged their work at the organization.
# - Most of the employees whose total business values is 5% are the highest no. of employees who are leaving the organization.
# - Most of the employees are getting low rating may be could be the reason of leaving the organization.
# - Very few employees are getting acknowledged their work at the organization and getting good rating.
# 
# - F1 Score => 85.5 and Accuracy (for Random Forest) => 76.84. And Score (for Boosting) => 76.97 
# 
# - AOC-RUC for Random Forest from Estimators => 0.78 , and from Predictions => 0.66.

# # Recommendations : 

# - Months Worked is one of the important feature, also from the graphical analysis its evident that the atleast 75 percentile of drivers that have left the organization have worked for 30 months or less. So if policies and arrangements can make sure that drivers stay upto 30-36 months after joining then its likely that they will be part of the organization for longer period of time. Also while hiring if we have an option of choosing between two candidates then its no brainer to pick up the candidate having higher average tenure at past organizations (given that this data is available).
# 
# - Average Monthly Income is also one of the important feature, hence good compensation model could ensure longer stay of the drivers, or else having some kind of bond for lower compensation driver could ensure their stay.
# 
# - Quarterly Rating and Business Value were highly correlated and hence developing a metric to monitor the drop of these values beyond a certain threshold for a particular driver could mean that there is some issue with the particular driver and hence it could be further discussed and any help needed could be extended to the extent possible, and this could help in decreasing attrition.
# 
# - City and Age also appears as important features. City for example could mean that bigger the city, better the opportunities from various ride business, again some sort of better policies could help decrease the attrition.

# In[ ]:




