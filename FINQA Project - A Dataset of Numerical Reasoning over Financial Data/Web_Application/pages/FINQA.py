#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install streamlit


# In[2]:


# !pip install google.generativeai


# In[3]:


# !pip install -q openai
# !pip install openai


# In[4]:


import numpy as np
import pandas as pd
import flask
import streamlit as st
import re
import google.generativeai as palm
import openai

# In[5]:


openai.api_key = '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'


# In[6]:


CONTENT = f"""Based on the Context provided, consisting of the Pre-Text , Tabular Data , and the Post-Text Data, answer the 
Financial Question mentioned below. Since calculating Financial Answers involve Numerical Computations, there are a maximum of 
6 Mathematical Operations that should be used : Add , Subtract , Multiply , Divide , Greater , Exponential. And a maximum of 4 
Table Aggregation Operations that should be used : Table Maximum , Table Minimum , Table Sum , Table Average.
The 6 Mathematical Operations are for the Unstructured Text Data , and the 4 Table Aggregation Operations are for the Structured
Tabular Data.

Also, along with answering the Financial Question mentioned, provide a complete 
Explanation to understand how to reach at this Solution of the Financial Question generated. Means that in First Line, generate 
only the numerical answer, and then in the next line, generate the explanation.

Note: While generating answers, look carefully at the numbers which are present in the Structured Tabular Data, and do not get
confused between the symbols like decimal(".") and comma(","). For example: 7,832 should be considered as 7832 and not 7.832.
Be strict on noticing whether any number has decimal or comma, as it could change the answer a lot. 

Context : 
Question: 
 
Answer :
Explanation : 

"""


# In[7]:


def generate_financial_answer_and_explanation(context,question):
    answer = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": CONTENT},
    ],
    temperature=0.6,
)
    return answer.choices[0].message.content


# In[8]:


st.set_page_config(
    page_title="Wallet - Your Financial Question Answering Assistant",
    layout='centered',
    initial_sidebar_state='collapsed')

def answer():
    st.title("Wallet - Your Financial Question Answering Assistant")
    st.header("Give me Texts & Tables, You get Numbers!!")

    st.markdown(
    '''
    <style>
        .center-image {
            display: flex;
            justify-content: center;
        }
    </style>
    <div class="center-image">
    </div>
    </a>
    <p></p>
    <p></p>
    <body>
        <header>
            <div>
                <div class="center-image">
                <h1>📈🏦📉</h1>
                </div>
            </div>
        </header>
    </body>
    ''',
    unsafe_allow_html=True)

    audio_file = open('Wallet.mp3', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')

    context = st.text_area("Enter the Context from the Report: ")
    question = st.text_area("Enter your Question: ")

    if st.button("Generate"):
        answer = generate_financial_answer_and_explanation(context,question)
        
        st.header("Answer: ")
        st.write(answer)
        
if __name__ == "__main__":
    answer()



