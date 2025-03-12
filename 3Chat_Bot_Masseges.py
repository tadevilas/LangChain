from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import SystemMessages , HumanMessages, AIMessages
import os
load_dotenv()



HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if HUGGINGFACEHUB_API_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
else:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found!")

llm = HuggingFaceEndpoint(repo_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', task = 'text-generation')


model  = ChatHuggingFace(llm= llm)

messages = [
    SystemMessages(content = 'You are helpful Assistant'),
    HumanMessages(content = 'Tell me about LangChain')

]

