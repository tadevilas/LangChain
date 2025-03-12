from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if HUGGINGFACEHUB_API_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
else:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found!")

llm = HuggingFaceEndpoint(repo_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', task = 'text-generation')


model  = ChatHuggingFace(llm= llm)


st.header('Research Tool')

user_input = st.text_input('Enter Your Prompt')

if st.button('Summarize'):
    if user_input:  # Ensure input is not empty
        result = model.invoke(user_input)
        st.write(result.content)  # Accessing the result from the response
    else:
        st.warning("Please enter a prompt to summarize.")


