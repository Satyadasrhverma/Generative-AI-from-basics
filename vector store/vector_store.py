from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# ✅ LLM (for answering)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# ✅ Embedding model (for vectors)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Documents
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history.",
        metadata={"team": "RCB"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history.",
        metadata={"team": "MI"}
    ),
    Document(
        page_content="MS Dhoni is famously known as Captain Cool and a legendary finisher.",
        metadata={"team": "CSK"}
    ),
    Document(
        page_content="Jasprit Bumrah is one of the best fast bowlers in T20 cricket.",
        metadata={"team": "MI"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder and bowler.",
        metadata={"team": "CSK"}
    )
]

# ✅ Vector store
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="my_chroma_db",
    collection_name="sample"
)

# Search
results = vector_store.similarity_search(
    query="who among these are a bowler",
    k=2
)

for r in results:
    print(r.page_content, r.metadata)
