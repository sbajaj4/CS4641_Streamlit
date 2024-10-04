import streamlit as st
import pandas as pd

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
The dataset used is available on Kaggle and includes 1,000 samples with customer data:
-CustomerID: Unique identifier.
-Age: Customer's age.
-Gender: Male/Female
-Tenure: Number of months with the service provider.
-MonthlyCharges: Monthly fees paid by the customer.
-ContractType: Month-to-Month, One-Year, or Two-Year.
-Churn: Target variable indicating whether the customer has churned (Yes/No).
"""
)
st.header('Problem Definition')
st.subheader('Problem')
st.subheader('Motivation')
st.header('Methods')
st.subheader('Data Preprocessing')
st.subheader('ML Models')
st.header('Potential Results')
st.subheader('Quantitative Metrics')
st.subheader('Project Goals')
st.subheader('Expected Results')
st.header('References')
st.header('Gannt Chart')
st.header('Contribution Table')
st.header('Github Repository')
# ------ PART 1 ------

df = pd.read_csv("customer_churn_data.csv")

# Display text
st.text('Fixed width text')
st.markdown('_**Markdown**_') # see #*
st.caption('Balloons. Hundreds of them...')
st.latex(r''' e^{i\pi} + 1 = 0 ''')
st.write('Most objects') # df, err, func, keras!
st.write(['st', 'is <', 3]) # see *
st.title('My title')
st.header('My header')
st.subheader('My sub')
st.code('for i in range(8): foo()')

# * optional kwarg unsafe_allow_html = True

# Interactive widgets
st.button('Hit me')
st.data_editor(df)
st.checkbox('Check me out')
st.radio('Pick one:', ['nose','ear'])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiselect', [1,2,3])
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
st.file_uploader('File uploader')

# ------ PART 2 ------

# Display Data
st.dataframe(df)
st.table(df.iloc[0:10])
st.json({'foo':'bar','fu':'ba'})
st.metric('My metric', 42, 2)

# Display Charts
st.area_chart(df[:10])
st.bar_chart(df[:10])
st.line_chart(df[:10])
# st.map(df[:10])
st.scatter_chart(df[:10])

# Add sidebar
a = st.sidebar.radio('Select one:', [1, 2])
st.sidebar.caption("This is a cool caption")

# Add columns
col1, col2 = st.columns(2)
col1.write("This is column 1")
col2.write("This is column 2")
