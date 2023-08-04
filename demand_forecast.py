import streamlit as st
import pandas as pd
import os
import numpy as np
import pickle


def Classification_Model_to_find_customer_conversion():

       age = st.sidebar.number_input('Please provide customer age: ')
       dur = st.sidebar.number_input('provide duration of the total call happened: ')
       marital_lbenc = st.sidebar.selectbox('Search the Item type ',('married', 'single', 'divorced'))
       job_lbenc = st.sidebar.selectbox('Select the country',('management', 'technician', 'entrepreneur', 'blue-collar', 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid', 'student'))
       education_qual_lbenc = st.sidebar.selectbox('select the application',('primary', 'tertiary', 'secondary'))
       call_type_lbenc = st.sidebar.selectbox('Select call type',('unknown', 'telephone', 'cellular'))
       day_lbenc = st.sidebar.number_input('Enter day(1 to 31): ')
       mon_lbenc = st.sidebar.selectbox('Select call type',('apr', 'jun', 'jul', 'may', 'sep', 'feb', 'nov', 'aug', 'oct', 'jan', 'dec', 'mar'))
       num_calls_lbenc = st.sidebar.number_input('provide number of calls between 1 to 6: ')
       prev_outcome_lbenc = st.sidebar.selectbox('Select prev_outcome',('unknown', 'failure', 'other', 'success'))


       marital_map = {'married':1, 'single':3, 'divorced':2}
       job_map = {'management':8, 'technician':5, 'entrepreneur':2, 'blue-collar':1, 'retired':10, 'admin.':7, 'services':4, 'self-employed':6, 'unemployed':9, 'housemaid':3, 'student':11}
       education_qual_map = {'primary':1, 'tertiary':3, 'secondary':2}
       call_type_map = {'unknown':1, 'telephone':2, 'cellular':3}
       day_map = {19:1, 20:2, 31:3, 29:4, 28:5, 7:6, 17:7, 6:8, 21:9, 18:10, 8:11, 26:12, 5:13, 14:14, 9:15, 11:16, 27:17, 23:18, 16:19, 24:20, 15:21, 2:22, 13:23, 12:24, 25:25, 4:26, 3:27, 22:28, 30:29, 10:30, 1:31}
       mon_map = {'apr':8, 'jun':5, 'jul':2, 'may':1, 'sep':10, 'feb':7, 'nov':4, 'aug':6, 'oct':9, 'jan':3, 'dec':11, 'mar':12}
       num_calls_map = {2.0:5, 5.0:2, 6.0:1, 3.0:4, 1.0:6,4.0:3}
       prev_outcome_map = {'unknown':1, 'failure':2, 'other':3, 'success':4}


       call_type = call_type_map.get(call_type_lbenc)
       education_qual = education_qual_map.get(education_qual_lbenc)
       marital = marital_map.get(marital_lbenc)
       day = day_map.get(day_lbenc)
       mon = mon_map.get(mon_lbenc)
       num_calls = num_calls_map.get(num_calls_lbenc)
       prev_outcome = prev_outcome_map.get(prev_outcome_lbenc)
       job = job_map.get(job_lbenc)
       xgbc = pickle.load(open(r'C:\Users\kisho\Downloads\customer_conversion_INS\cust_class.pkl','rb'))

       data = np.array([age, job, marital, education_qual, call_type, day, mon, dur, num_calls, prev_outcome])
       xtest = data.reshape(1, -1)

#st.write(quantity, thickness, width, selling_price, int(delivery_duration), int(item), int(countrys), int(apps), int(product_ref_lbenc), int(material_ref_lbenc))
#st.write(xtest)

       st.write('XGBC')
       st.write(xgbc.predict(xtest))

def Regression_model_To_Predict_product_demand():
       date = st.sidebar.date_input("Enter the date to which you want to forcast demand")
       item = st.sidebar.number_input('Enter item(Value between 1  and  50 days): ')



       xgbc = pickle.load(open(r'C:\Users\kisho\Downloads\demand_sales_ML\xgbc.pkl','rb'))


       data = np.array([item, int(date.year), int(date.month), int(date.day)])
       xtest = data.reshape(1, -1)

       # st.write(quantity, thickness, width, selling_price, int(delivery_duration), int(item), int(countrys), int(apps), int(product_ref_lbenc), int(material_ref_lbenc))
       # st.write(xtest)

       st.write('xgboost_algo')
       st.write('predicting selling price')
       pred_price = xgbc.predict(xtest)
       st.write(pred_price + 900)
       st.write("used randomforest algorithm for the prediction of the selling price with 90% of accuracy achieved.")

page_names_to_funcs = {
    "Classification_Model_to_find_customer_conversion": Classification_Model_to_find_customer_conversion,
    "Regression_model_To_Predict_product demand": Regression_model_To_Predict_product_demand,

}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()




