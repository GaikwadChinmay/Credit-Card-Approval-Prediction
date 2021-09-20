# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:11:03 2021

@author: ChiGa
"""

import streamlit as st
import pandas as pd
#from sklearn import datasets
#from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

#Model
# Loading dataset
url = "https://github.com/GaikwadChinmay/Classification-CreditCardApproval/blob/main/cc_approvals.data"
cc_apps = pd.read_csv(url)
cc_apps_labelled = cc_apps.rename(columns={0:'Gender', 1:'Age', 2:'Debt', 3:'Married', 4:'BankCustomer', 5:'EducationLevel', 6:'Ethnicity', 7:'YearsEmployed', 8:'PriorDefault', 9:'Employed', 10:'CreditScore', 11:'DriversLicense', 12:'Citizen', 13:'ZipCode', 14:'Income',15:'ApprovalStatus'})
# Inspecting data
cc_apps_labelled = cc_apps_labelled.replace('?',np.NaN)
cc_apps_labelled['Gender'] = cc_apps_labelled['Gender'].map({'b':'M','a':'F'})
cc_apps_labelled['Married'] = cc_apps_labelled['Married'].map({'u':'N','y':'Y','l':'D'})
cc_apps_labelled['BankCustomer'] = cc_apps_labelled['BankCustomer'].map({'g':'Y','p':'N','gg':'N'})
cc_apps_labelled['EducationLevel'] = cc_apps_labelled['EducationLevel'].map({'c':'Graduate','q':'PostGraduate','w':'PGDiploma','i':'Diploma','aa':'Graduate','ff':'HighSchool','k':'PhD','cc':'Secondary','m':'Primary','x':'Primary','d':'Secondary','e':'HighSchool','j':'Graduate','r':'Secondary'})
cc_apps_labelled['Ethnicity'] = cc_apps_labelled['Ethnicity'].map({'v':'Asian','h':'African','bb':'LatinAmerican','ff':'European','j':'Oceania','z':'MiddleEast','dd':'Latin','n':'EastAsian','o':'SouthAmerican'})
cc_apps_labelled['PriorDefault'] = cc_apps_labelled['PriorDefault'].map({'f':'N','t':'Y'})
cc_apps_labelled['Employed'] = cc_apps_labelled['Employed'].map({'f':'N','t':'Y'})
cc_apps_labelled['Age'] = cc_apps_labelled['Age'].astype(float)
cc_apps_labelled['Citizen'] = cc_apps_labelled['Citizen'].map({'g':'Y','s':'N','p':'Refugee'})
cc_apps_labelled['ApprovalStatus'] = cc_apps_labelled['ApprovalStatus'].map({'+':'Y','-':'N'})
cc_apps_labelled['DriversLicense'] = cc_apps_labelled['DriversLicense'].map({'f':'N','t':'Y'})
# Imputing the missing values with mean imputation
cc_apps_labelled.select_dtypes(['int','float']).fillna(cc_apps_labelled.mean(), inplace=True)
# Iterating over each column of cc_apps
for col in cc_apps_labelled.columns:
    # Checking if the column is of object type
    if cc_apps_labelled[col].dtype == 'object':
        # Imputing with the most frequent value
        cc_apps_labelled = cc_apps_labelled.fillna(max(cc_apps_labelled[col].value_counts()))
cc_apps_labelled.drop(labels='ZipCode',axis=1,inplace=True)
cols1 = cc_apps_labelled.select_dtypes(np.object).columns.tolist()
# Feature encoding
from sklearn.preprocessing import LabelEncoder
# Instantiating LabelEncoder
le=LabelEncoder()
# Iterating over all the values of each column and extract their dtypes
for col in cc_apps_labelled.columns.to_numpy():
    # Comparing if the dtype is object
    if cc_apps_labelled[col].dtypes=='object':
        #print(cc_apps[col])
    # Using LabelEncoder to do the numeric transformation
        cc_apps_labelled[col]=le.fit_transform(cc_apps_labelled[col].astype(str))
# Importing train_test_split
from sklearn.model_selection import train_test_split
# Segregating features and labels into separate variables
X,y = cc_apps_labelled.drop(labels='ApprovalStatus',axis=1) , cc_apps_labelled['ApprovalStatus']
# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)
# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)
# Importing LogisticRegression
from sklearn.linear_model import LogisticRegression
# Instantiating a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()
# Fitting logreg to the train set
final_logreg = LogisticRegression(tol= 0.01,max_iter=100)
final_logreg.fit(rescaledX_train,y_train)
# Model Complete


# Streamlit
st.write("""
# Credit Card Approval Prediction
This web app predicts the **Credit Card Approval rate**
""")

st.sidebar.header('User Input Parameters')

def _max_width_(prcnt_width:int = 75):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )
    
def user_input_features():
    Gender = st.sidebar.selectbox('Gender',('Male', 'Female'))
    Age = st.sidebar.slider('Age (In Years)', 10, 80, 1)
    Income = st.sidebar.slider('Income (In Million)', 0,10, 1)
    Debt = st.sidebar.slider('Debt (In Million)', 0,30, 1)
    YearsEmployed = st.sidebar.slider('Years Employed (In Years)', 1,30, 1)
    CreditScore = st.sidebar.slider('Credit Score', 0,20, 1)
    MaritalStatus = st.sidebar.selectbox('Marital Status',('Married', 'Bachelore','Divorced'))
    BankCustomer = st.sidebar.selectbox('Bank Customer',('Yes', 'No'))
    Education = st.sidebar.selectbox('Education',('Graduate', 'PostGraduate','HighSchool','Primary','PGDiploma','Diploma','PhD','Secondary'))
    Ethinicity = st.sidebar.selectbox('Ethinicity',('Asian', 'African','EastAsian','SouthAmerican','LatinAmerican','European','Oceania','MiddleEast','Latin'))
    PriorDefault = st.sidebar.selectbox('Prior Default',('Yes', 'No'))
    DriversLicense = st.sidebar.selectbox('Drivers License',('Yes', 'No'))
    Citizen = st.sidebar.selectbox('Citizen',('Yes', 'No','Refugee'))
    Employed = st.sidebar.selectbox('Employed',('Yes', 'No'))
    data = {'Gender': Gender,
            'Age': Age,
            'Debt': Debt,
            'MaritalStatus': MaritalStatus,
            'BankCustomer':BankCustomer,
            'Education':Education,
            'Ethinicity':Ethinicity,
            'YearsEmployed':YearsEmployed,
            'PriorDefault':PriorDefault,
            'Employed':Employed,
            'CreditScore':CreditScore,
            'DriversLicense':DriversLicense,
            'Citizen':Citizen,
            'Income':Income
            }
    features = pd.DataFrame(data, index=[0])
    return features


df_inp = user_input_features()
st.subheader('User Input parameters')
st.write(df_inp)

for col in df_inp.columns.to_numpy():
    # Comparing if the dtype is object
    if df_inp[col].dtypes=='object':
        #print(cc_apps[col])
    # Using LabelEncoder to do the numeric transformation
        df_inp[col]=le.fit_transform(df_inp[col].astype(str))



prediction = final_logreg.predict(df_inp)
prediction_proba = final_logreg.predict_proba(df_inp)

#st.subheader('Class labels and their corresponding index number')
target_names=['Approved','Not Apporved']
pred = [1-prediction, prediction]
prob = prediction_proba
#st.write(target_names,index=0)
st.subheader('Prediction Probability')
output_proba = pd.DataFrame(np.array(prob), columns=['Not Approved','Approved'])
st.write(output_proba)
#st.write(prediction_proba)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
#st.subheader('Prediction Probability')
labels = target_names
sizes = (prob[0][1],prob[0][0])
colors = ['mediumseagreen', 'tab:red']
explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
fig1, ax1 = plt.subplots(constrained_layout=True,figsize=(6 ,2))


ax1.pie(sizes, explode=explode, labels=labels,colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90,textprops={'fontsize': 9},pctdistance=0.45,labeldistance=1.2)
ax1.legend(('Approved','Not Apporved'), loc='upper right', shadow=True,prop={'size':5.5})
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)

output = pd.DataFrame(data = {'Not Approved':1-prediction,'Approved':prediction},index=[0])
st.subheader('Prediction')
st.write(output)
#st.write(prediction)
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
st.text('Dev: Chinmay Gaikwad')
st.text('Email: chinmaygaikwad123@gmail.com')

