from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1️⃣ Load PDF
loader = PyPDFLoader("RAG Components/PDF loader/ai_full_notes_7_pages.pdf")
docs = loader.load()

# 2️⃣ Combine ALL pages
full_text = "\n".join(doc.page_content for doc in docs)

# 3️⃣ Load LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# 4️⃣ Prompt (FIXED)
prompt = PromptTemplate(
    template="""
Create exactly 5 multiple-choice questions (MCQs) with answers
based ONLY on the content below.

Format:
Q1. Question
A. option
B. option
C. option
D. option
Answer: X

Content:
{text}
""",
    input_variables=["text"]
)

# 5️⃣ Output parser
parser = StrOutputParser()

# 6️⃣ Chain
chain = prompt | model | parser

# 7️⃣ Run
result = chain.invoke({"text": full_text})

print(result)
