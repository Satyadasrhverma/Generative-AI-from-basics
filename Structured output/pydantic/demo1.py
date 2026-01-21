from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# LLM setup
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Define schema
class Review(BaseModel):
    summary: str
    sentiment: str

# Create parser
parser = PydanticOutputParser(pydantic_object=Review)

# Prompt with format instructions
prompt = f"""
Analyze the following review and respond ONLY in the format below.

{parser.get_format_instructions()}

Review:
The hardware is great, but the software feels bloated. There are too
many pre-installed apps that I can't remove. Also, the UI looks outdated
compared to other brands. Hoping for a software update to fix this.
"""

# Invoke model
response = model.invoke(prompt)

# Parse + validate
result = parser.parse(response.content)

print(result) 
