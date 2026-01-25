from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,SystemMessage, AIMessage

load_dotenv()

# 1️⃣ Load PDF once
loader = PyPDFLoader("RAG Components/PDF loader/ai_full_notes_7_pages.pdf")
docs = loader.load()
full_text = "\n".join(doc.page_content for doc in docs)

# 2️⃣ LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content=f"""
You are a helpful assistant.
Answer ONLY using the document below.
If the answer is not present, say "Not found in document".

Document:
{full_text}
""")
]

# 4️⃣ Chat loop
while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    chat_history.append(HumanMessage(content=user_input))

    result = model.invoke(chat_history)

    chat_history.append(AIMessage(content=result.content))

    print("AI:", result.content)