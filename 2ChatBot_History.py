from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
import os
load_dotenv()


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if HUGGINGFACEHUB_API_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
else:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found!")

llm = HuggingFaceEndpoint(repo_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', task = 'text-generation')


model  = ChatHuggingFace(llm= llm)

chat_history = []

while True:
    user_input = input('You: ')
    chat_history.append(user_input)

    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print('AI:', result.content)

print(chat_history)