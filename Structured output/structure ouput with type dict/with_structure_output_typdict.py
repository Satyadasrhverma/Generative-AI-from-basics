from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# LLM setup
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Define TypedDict schema (TYPE ONLY)
class Review(TypedDict):
    summary: str
    sentiment: str

# Create JSON parser
parser = JsonOutputParser()

# Prompt with format instructions
prompt = f"""
Analyze the following review and return ONLY valid JSON
with keys: summary and sentiment.

Review:
The hardware is great, but the software feels bloated. There are too
many pre-installed apps that I can't remove. Also, the UI looks outdated
compared to other brands. Hoping for a software update to fix this.
"""

# Invoke model
response = model.invoke(prompt)

# Parse JSON -> dict
result = parser.parse(response.content)

print(result)
