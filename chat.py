import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
# モデルはGemini-1.5-flashを指定
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

st.title("langchain-streamlit-app")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = llm.invoke(prompt)
        st.markdown(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})