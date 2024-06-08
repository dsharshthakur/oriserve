import streamlit as st
from main import UserSentiment
from dotenv import load_dotenv
import os
import re

# loading api key
load_dotenv()
google_key = st.secrets["PROJECT_KEY"]

# custom CSS for submit button
btn_style = '''<style>
.stButton button {
        background-color:#00CED1;
        color: black;
        transition: background-color 0.3s;
        
        }
.stButton button:hover {
    background-color:#4CE489;
    border : 2px solid yellow;
    color : black;
    }
    </style>
'''

st.markdown(btn_style, unsafe_allow_html=True)
st.markdown("<h2 style= 'text-align:center'>User Sentiment Analysis</h2>", unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

# take user input
review_text = st.text_area(label="Insert the text here..")


# layout for button
btn_area = st.container()
with btn_area:
    col1, col2, col3 = btn_area.columns(3)
    submit_btn = col2.button('Submit', use_container_width=True)


# if button is active
if submit_btn and review_text != "":
    # model object
    model = UserSentiment(api_key=google_key, llm_name="gemini-pro")

    # generating the AI - Response
    answer = model.generate(user_review=review_text)

    answer = re.sub(r'[\[\]]', "", string=answer).split(", ")

    # list containing positive response only
    positive_response = [txt for txt in answer if 'positive' in txt]

    # list containing negative response only
    negative_response = [txt for txt in answer if 'negative' in txt]

    # display response if positive response is not empty - Green color
    if len(positive_response) != 0:
        for sentiment in positive_response:
            st.success(sentiment)

    # display response if negative response is not empty - Red color
    if len(negative_response) != 0:
        for sentiment in negative_response:
            st.error(sentiment)

