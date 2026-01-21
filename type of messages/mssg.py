from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage , HumanMessage, AIMessage
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
) 

model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content= "you are a helpul assistant"),
    HumanMessage(content="tell me about langchain")
]

result =model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)