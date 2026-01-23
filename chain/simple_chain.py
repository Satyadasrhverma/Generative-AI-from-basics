from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# LLM setup
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template= "" 
        "Explain ChatGPT features in plain text only.Do NOT use markdown, bullet points, bold text (**), or formatting.\n\n"
        "Give 5 point of facts about {topic}.",
        input_variables= ['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic' : 'Chatgpt'})

print(result)