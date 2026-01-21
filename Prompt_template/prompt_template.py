from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Create LLM connection
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

# 2. Wrap LLM as chat model
model = ChatHuggingFace(llm=llm)

# 3. Create chat prompt template (CORRECT way)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful {domain} expert"),
    ("human", "explain in simple terms what is {topic}")
])

# 4. Fill variables in prompt
prompt = chat_template.invoke({
    "domain": "Cricket",
    "topic": "bowling"
})

# 5. Send prompt to model
response = model.invoke(prompt)

# 6. Print AI response
print(response.content)
