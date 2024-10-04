import streamlit as st
import pandas as pd

st.caption('Saumya Bajaj, Katniss Min, Liane Nguyen, Calvin Truong, Xiangyi Zhu')
st.title('Proposal')
st.header('Introduction')
st.write('Customer churn is an important issue in the telecom industry, leading to revenue loss and additional costs to acquire new customers. To tackle this, companies have started predicting customer churn by identifying those at risk. This project will aim to develop a machine learning model that accurately predicts customer churn, minimizing profit loss and improving customer retention.')
st.header('Background')
st.subheader('Literature Review')
st.write('The majority of the literature aims to predict whether a customer will churn. J. Bhattacharyya and M. K. Dash identified “ten overarching groups of scholarship” [1], the first two being churn prediction and modeling and feature selection techniques and comparison. Another paper proposes a six-step methodology starting with data pre-processing and feature analysis [2]. However, despite the extensive research and demand for churn reduction tools, there is a lack of “well-defined guidelines on appropriate model evaluation measures” [3].')
st.write('There is a great demand for cross-industry model evaluation as well as a more generalized churn prediction model, as most are performed on one specific industry or consumer base [1]-[3].')
st.subheader('Dataset Description')
st.markdown(
"""
The dataset used is available on Kaggle [4] and includes 1,000 samples with customer data:
- CustomerID: Unique identifier.
- Age: Customer's age.
- Gender: Customer's gender.
- Tenure: Number of months with service provider.
- MonthlyCharges: Monthly fees paid by customer.
- ContractType: Customer's contract type.
- InternetService: Type of internet service subscribed to.
- TechSupport: Whether customer has tech support.
- TotalCharges: Customer's total charges.
- Churn: Target variable indicating whether the customer has churned.
"""
)
