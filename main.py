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
df = pd.read_csv("customer_churn_data.csv")
st.header('Problem Definition')
st.subheader('Problem')
st.write('Customer churn negatively impacts telecom companies by reducing revenue and increasing customer acquisition costs. To mitigate this, telecom companies need a predictive model to identify customers likely to churn, allowing them to take preventive actions.')
st.subheader('Motivation')
st.write('Retaining customers is far more cost-effective than acquiring new ones. By identifying customers at risk of churning, companies can improve profitability through targeted retention strategies.')
st.header('Methods')
st.subheader('Data Preprocessing')
st.markdown(
"""
1. One-Hot Encoding: Convert categorical variables into numerical form.
2. MinMax Scaling: Normalize numeric features 
3. Imputation: Handle any missing data to ensure a complete dataset.
"""
)
st.subheader('ML Models')
st.markdown(
"""
1. Logistic Regression (Supervised):
  - Goal: Predict the probability of customer churn based on demographic and contract features.
  - Effectiveness:
    - Is a well established method for binary classification
    - Easy to interpret prediction results
    - Can process large volumes of data at a high speed
    - Supervised model makes sense for evaluation
2. Principal Component Analysis (PCA) (Unsupervised):
  - Goal: Reduce dimensionality and identify the most important components driving customer behavior.
  - Effectiveness: 
    - Reduce dimensionality for features
    - More computational effective by removing irrelevant noise data in variance, especially when working with K-Means Clustering.
    - Highlight important features
3. K-Means Clustering (Unsupervised):
  - Goal: Group customers into distinct clusters based on demographic and service usage patterns to identify high-risk customer segments.
  - Effectiveness: 
    - Able to identify similar patterns 
    - Creates new groups of clustered data could be used for prediction training

"""
)
st.header('Potential Results')
st.subheader('Quantitative Metrics')
st.markdown(
  """
  1. F1-Score: For evaluating classification performance in Logistic Regression, especially with imbalanced data.
  2. Silhouette Score: For assessing the quality of customer groupings in K-Means Clustering.
  3. Explained Variance: For measuring how much data variance is preserved in the PCA model.
  """
)
st.subheader('Project Goals')
st.markdown(
  """
  1. Develop accurate and interpretable models to identify customers at high risk of churn.
  2. Use PCA to simplify the dataset and highlight the most significant customer features for actionable insights.
  3. Promote sustainability by reducing churn, minimizing resource-intensive customer acquisition.
  4. Avoid bias in predictions, ensuring fair treatment of sensitive demographic features.
  """
)
st.subheader('Expected Results')
st.write("The logistic regression model expects an F1-score above 0.8 to predict likely churners. PCA will reduce the dataset’s dimensions by 50% while preserving 90% of the data's variance. K-Means will segment customers into 3-5 groups with a silhouette score above 0.6, identifying at-risk customer segments for targeted retention.")
st.header('References')
st.markdown('[1] J. Bhattacharyya and M. K. Dash, “What do we know about customer churn behaviour in the telecommunication industry? A bibliometric analysis of Research Trends, 1985–2019,” FIIB Business Review, vol. 11, no. 3, pp. 280–302, Dec. 2021. doi:10.1177/23197145211062687 ')
st.markdown('[2] P. Lalwani, M. K. Mishra, J. S. Chadha, and P. Sethi, “Customer churn prediction system: A machine learning approach,” Computing, vol. 104, no. 2, pp. 271–294, Feb. 2021. doi:10.1007/s00607-021-00908-y ')
st.markdown('[3] S. De, P. P, and J. Paulose, “Effective ML techniques to predict customer churn,” 2021 Third International Conference on Inventive Research in Computing Applications (ICIRCA), Sep. 2021. doi:10.1109/icirca51532.2021.9544785')
st.markdown('[4] M. Abdullah, “Customer churn prediction:Analysis,” Kaggle, https://www.kaggle.com/datasets/abdullah0a/telecom-customer-churn-insights-for-analysis. ')
st.header('Contribution Table')
cont = [
          ['Liane', 'Dataset description, problem definition, methods, preprocessing methods, models, quantitative metrics, project goals, expected results, gantt chart, presentation slides'],
          ['Saumya', 'Literature review, references, Streamlit page, GitHub Repo'],
          ['Katniss', 'Intro, models, presentation slides, video'],
          ['Calvin', 'Machine Learning models, Data processing models, References, presentations'],
          ['Echo', 'Models, references, presentations']
        ]
c = pd.DataFrame(cont, columns=['Name', 'Proposal Contributions'])
st.table(c)
st.link_button('Dataset', 'https://www.kaggle.com/datasets/abdullah0a/telecom-customer-churn-insights-for-analysis')
st.link_button('Gantt Chart', 'https://docs.google.com/spreadsheets/d/e/2PACX-1vR4m7AUYW1z9pl9Z38wzCu2Xv63LlI7Fedj6opTo7QdBGat9jsuPFo4QRQ0VApDbw/pubhtml')
st.link_button('Github Repo', 'https://github.gatech.edu/sbajaj43/cs4641project')
