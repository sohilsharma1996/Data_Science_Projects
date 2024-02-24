import streamlit as st

base="light"
primaryColor="#ffdc4b"
secondaryBackgroundColor="#4b8eaf"
font="serif"


st.title("FINQA : A Dataset of Numerical Reasoning over Financial Data")
st.image("stocks.jpg", width=200, use_column_width=True)
st.sidebar.image("financenlp.jpg", use_column_width=False)
st.header("Problem Statement:")

st.text("""1. Financial Analysis is a critical means to assess Business Performance 
and the Consequences of Poor Analysis can involve costing Billions of Dollars to 
facilitate High-Quality Time and Decision-Making Professionals such as Analysts or 
Investors will perform Complex Quantity Analysis to select information from 
Financial Reports and such Analysis.
To facilitate Analytical Progress, we propose a new Large-Scale Dataset, FINQA, 
with Question-Answering Pairs over Financial Reports, written by Financial Experts.

2. There are 6 mathematical operations: add, subtract, multiply, divide,
greater, exp, and 4 table aggregation operations: table-max, table-min, table-sum,
table-average, that apply aggregation operations on table rows. The mathematical 
operations take arguments of either numbers from the given reports, or a numerical 
result from a previous step; The table operations take arguments of table row names. """)

st.image("operations.jpg", width=200, use_column_width='always')

st.text("""3. The sheer volume of financial statements makes it difficult for humans to access 
and analyze a business's financials. Robust numerical reasoning likewise faces 
unique challenges in this domain. In this work, we focus on answering deep questions 
over financial data, aiming to automate the analysis of a large corpus of financial 
documents. In contrast to existing tasks on general domain, the finance domain 
includes complex numerical reasoning and understanding of heterogeneous 
representations. Here, given Input Data in the form of Structured Tables and 
Unstructured Text , we need to find out the Solutions as per the Question asked , 
as per the Mathematical Calculations.""")
        
st.header("Some Links:") 
st.write("GITHUB LINK: [https://github.com/czyssrs/FinQA/tree/main](https://github.com/czyssrs/FinQA/tree/main)")
st.write("KAGGLE LINK: [https://www.kaggle.com/datasets/visalakshiiyer/question-answering-financial-data](https://www.kaggle.com/datasets/visalakshiiyer/question-answering-financial-data)")
st.write("Papers with Code LINK: [https://paperswithcode.com/paper/finqa-a-dataset-of-numerical-reasoning-over](https://paperswithcode.com/paper/finqa-a-dataset-of-numerical-reasoning-over)")
st.write("HTML Page for an Example: [https://finqasite.github.io/explore.html](https://finqasite.github.io/explore.html)")
st.write("CodaLab Competitions Page: [https://codalab.lisn.upsaclay.fr/competitions/1846#learn_the_details](https://codalab.lisn.upsaclay.fr/competitions/1846#learn_the_details)")
st.write("My Project Link: [https://encr.pw/UWoxc](https://github.com/sohilsharma1996/Data_Science_Projects/tree/main/FINQA%20Project%20-%20A%20Dataset%20of%20Numerical%20Reasoning%20over%20Financial%20Data)")

st.header("Objectives of this Project: ")
st.text("""
1. Perform Exploratory Data Analysis.
2. Build Model which can be used for Q/A.
3. Build a Web Application.
4. Deployment of the Model
        """)

   
