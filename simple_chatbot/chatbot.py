from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
) 

model = ChatHuggingFace(llm = llm)

while True:
    user_input = input("You:")
    if user_input == "exit" :
        break

    result = model.invoke(user_input)
    print("AI:" , result.content)