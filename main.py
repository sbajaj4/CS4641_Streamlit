import streamlit as st
import pandas as pd

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
