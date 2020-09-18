#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#Scipy Library
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


# In[2]:


patient_stay = pd.read_csv("train.csv")
train = patient_stay
train


# In[3]:


# Python Method 1 : Displays Data Information :

def display_data_information(data , data_types ,dataframe_name):
    print("Information of ",dataframe_name," : Rows = ",data.shape[0] , " | Columns = ",data.shape[1],"\n")
    data.info()
    print("\n")
    for VARIABLE in data_types :
        data_type = data.select_dtypes(include = [VARIABLE]).dtypes
        if len(data_type) > 0 :
            print(str(len(data_type)) + " " + VARIABLE + " Features\n" + str(data_type)+"\n")


# In[ ]:





# In[4]:


# Display Data Information of "patient_profile" :

data_types  = ["float32","float64","int32","int64","object","category","datetime64[ns]"]
display_data_information(patient_stay, data_types, "patient_length_of_stay")


# In[5]:


# Python Method 2 : Displays Data Head (Top Rows) and Tail (Bottom Rows) of the Dataframe (Table) :

def display_head_tail(data, head_rows, tail_rows):
    display("Data Head & Tail :")
    display(data.head(head_rows).append(data.tail(tail_rows)))
#     return True

# Displays Data Head (Top Rows) and Tail (Bottom Rows) of the Dataframe (Table)
# Pass Dataframe as "patient_stay", No. of Rows in Head = 3 and No. of Rows in Tail = 2 :

display_head_tail(patient_stay, head_rows=3, tail_rows=2)


# In[6]:


# Python Method 3 : Displays Data Description using Statistics :
def display_data_description(data , numeric_data_types = None , categorical_data_types = None):
    print("Data Description :")
    display(data.describe(include = numeric_data_types))
    print("")
    display(data.describe(include = categorical_data_types))

# Displays Train Data Description
display_data_description(patient_stay , data_types[0:4] , data_types[4:7])


# In[7]:


#Python Method 4 : Remove Data Duplicates while Retaining the First one - Similar to SQL DISTINCT :

def remove_duplicate(data):
    
    print("BEFORE REMOVING DUPLICATES - No. of Rows = ",data.shape[0])
    data.drop_duplicates(keep = "first" , inplace =True)
    print("AFTER REMOVING DUPLICATES  - No. of Rows = ",data.shape[0]) 
    return data


# In[8]:


# Remove Duplicates from "train" data
train = remove_duplicate(train)


# In[9]:


# Python Method 5 : Fills or Imputes Missing values with Various Methods : 

def fill_missing_values(data, fill_value, fill_types, columns, dataframe_name):
    
    print("Missing Values BEFORE REMOVAL in ",dataframe_name," data")
    display(data.isnull().sum())
    for column in columns :
        
        # Fill Missing Values with Specific Value :
        if "Value_Fill" in fill_types :
            data[column].fillna(fill_value , inplace = True)
           #print("Value_Fill")

        # Fill Missing Values with Forward Fill  (Previous Row Value as Current Row in Table) :
        if "Forward_Fill" in fill_types :
            data[ column ] = data[ column ].ffill(axis = 0)
#             print("Forward_Fill")

        # Fill Missing Values with Backward Fill (Next Row Value as Current Row in Table) :
        if "Backward_Fill" in fill_types :
            data[ column ] = data[ column ].bfill(axis = 0)
#             print("Backward_Fill")
    
    print("Missing Values AFTER REMOVAL in ",dataframe_name," data")
    display(data.isnull().sum())
    
    return data


# In[10]:


#Filling the missing values in City Code Patient

fill_value = stats.mode(train["City_Code_Patient"] , axis =None)
fill_value = int(np.squeeze(fill_value[0])  )
print(fill_value)
fill_types = [ "Value_Fill"]
#fill_missing_values(train , fill_value ,fill_types, "City_Code_Patient" , "train")

train["City_Code_Patient"].fillna(fill_value , inplace = True)
train.isnull().sum()


# In[11]:


train['Bed Grade'].unique()
null_indices_of_bedgrade = train[train['Bed Grade'].isnull()].index.tolist()


# In[12]:


#getting uniques value of hospital code w.r.to null values of bed grade to list
list1 = train.iloc[null_indices_of_bedgrade]['Hospital_code'].unique()
list1 = list(list1)
for i in list1:
    m = train['Hospital_code'] == i
    mode_value = int(stats.mode(train.loc[m , 'Bed Grade'])[0])  
    train.loc[m , 'Bed Grade'] = train.loc[m , 'Bed Grade'].fillna(mode_value)


# In[13]:


# Python Method 6 : Displays Unique Values in Each Column of the Dataframe(Table) :

def display_unique(data):
    for column in data.columns :
        
        print("No of Unique Values in "+column+" Column are : "+str(data[column].nunique()))
        print("Actual Unique Values in "+column+" Column are : "+str(data[column].sort_values(ascending=True,na_position='last').unique() ))
        print("NULL Values :")
        print(data[ column ].isnull().sum())
        print("Value Counts :")
        print(data[column].value_counts())
        print("")


# In[14]:


#Renameing some column
train=train.rename(columns={"Severity of Illness": "Severity_of_Illness", "Type of Admission": "Type_of_Admission"})


# In[15]:


train.isnull().sum()


# In[16]:


train=train.drop(["patientid","case_id"],axis=1)


# In[17]:


columns=list(train.columns)
columns
unique=[]
for col in columns:
    uni=train[col].nunique()
    unique.append(uni)
    print(col,uni)


# In[18]:


train["Stay"].unique()


# In[19]:


test=pd.read_csv("/home/akaul/Desktop/hackthon/Test_ND2Q3bm/test.csv")
test
test["case_id"]=


# In[20]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
train['Stay'] = le.fit_transform(train['Stay']).astype(int)


# In[21]:


#lable encoding of age
Age_ordinal={ "0-10":0,
              '11-20':1,
              '21-30':2, 
              '31-40':3, 
              '41-50':4,
              '51-60':5,
              '61-70':6,
              '71-80':7,
              '81-90':8,
              '91-100':9,
}
train["Age"]=train.Age.map(Age_ordinal)
train["Age"]=train["Age"].astype("category")


# In[22]:


train["Stay"]


# In[23]:


#lable encoding of severity
Severity={"Moderate":0,
          "Extreme":1,
          "Minor":2,
}
train["Severity_of_Illness"]=train.Severity_of_Illness.map((Severity))
train["Severity_of_Illness"]=train["Severity_of_Illness"].astype("category")


# In[24]:


#lable encoding of age Department
department={'gynecology':0,
            'anesthesia':1,
            'TB & Chest disease':2,
            'radiotherapy':3,
            'surgery':4
           }
train["Department"]=train.Department.map(department)
train["Department"]=train["Department"].astype("category")


# In[25]:


#lable encoding of addmission
addmission={'Emergency':1,
           'Trauma':2,
            'Urgent':0
}
train["Type_of_Admission"]=train.Type_of_Admission.map(addmission)
train["Type_of_Admission"]=train["Type_of_Admission"].astype("category")


# In[26]:


train.info()


# In[27]:


category_train=train.columns[train.dtypes=="category"].tolist()
object_train=train.columns[train.dtypes==object].tolist()
category_train=category_train+object_train
cg_features = list(train[category_train].columns)
sp=1
for columns in cg_features:
    plt.subplot(4,4,sp)
    plt.title(columns)
    plt.hist(train[columns])  
    sp+=1
fig = plt.gcf()
fig.set_size_inches(20.5, 20.5)
plt.tight_layout()


# In[28]:


df_test=pd.read_csv("df_test.csv")


# In[29]:


final_df=pd.concat((train,df_test),axis=0)
final_df["Age"]=final_df["Age"].astype("category")
final_df["Severity_of_Illness"]=final_df["Severity_of_Illness"].astype("category")
final_df["Department"]=final_df["Department"].astype("category")
final_df["Type_of_Admission"]=final_df["Type_of_Admission"].astype("category")
final_df["Bed Grade"]=final_df["Bed Grade"].astype("category")
final_df["Stay"]=final_df["Stay"].fillna(0)
final_df["Stay"]=final_df["Stay"].astype(int)


# In[30]:


final_df.info()


# In[31]:


object_final_df=final_df.columns[final_df.dtypes==object].tolist()
category_=final_df.columns[final_df.dtypes=="category"].tolist()
cols=category_train+object_train
cols


# In[32]:


final_df=pd.get_dummies(final_df)


# In[33]:


final_df.info()


# In[34]:


final_df=final_df.loc[:,~final_df.columns.duplicated()]
final_df.shape


# In[35]:


test_x=final_df.iloc[318438:,:]
test_x=test_x.drop("Stay",axis=1)
test_x


# In[36]:


train_x=final_df.iloc[:318437,:]
#train_x["Stay"]=train_x["Stay"].astype(int)
train_x["Stay"]


# In[37]:


X_train=train_x.drop(["Stay"],axis=1)
X_train


# In[38]:


Y_train=(train_x["Stay"])
Y_train


# In[39]:


from  sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()


# In[40]:


dtc.fit(X_train,Y_train)


# In[41]:


y_predict1=dtc.predict(test_x)
pd.set_option('display.max_rows',318436)
y_predict1


# In[42]:


preds = le.inverse_transform(y_predict1)
preds


# In[54]:


predict=pd.DataFrame(y_predict1)
sub_df=pd.read_csv("sample_submission_lfbv3c3.csv")
datasets=pd.concat([sub_df["case_id"],predict],axis=1)
datasets.columns=["case_id","Stay"]
datasets.to_csv("sample_submission_lfbv3c3.csv",index=False)


# In[55]:


df_sub=pd.read_csv("sample_submission_lfbv3c3.csv")
df_sub


# In[ ]:




