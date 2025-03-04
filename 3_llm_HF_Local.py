from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set Hugging Face cache directory (optional)
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# Initialize HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100  # Corrected to 'max_new_tokens'
    )
)

# Initialize the model using the HuggingFace pipeline
model = ChatHuggingFace(llm=llm)

results = model.invoke('What is capital of India')

print(results.content)