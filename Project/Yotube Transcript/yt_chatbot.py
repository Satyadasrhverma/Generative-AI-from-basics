from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

video_id = "Gfr50f6ZBvo"


try:
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    transcript = transcript_list.find_transcript(
        ["en", "en-US", "a.en"]
    ).fetch()

    transcript = " ".join(item["text"] for item in transcript)

except TranscriptsDisabled:
    print("❌ Transcripts are disabled for this video.")
    exit()

except NoTranscriptFound:
    print("❌ No English transcript found.")
    exit()

except Exception as e:
    print("❌ Failed to fetch transcript:", e)
    exit()


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.create_documents([transcript])

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(chunks,embedding)

retriver = vector_store.as_retriever(search_type ="similarity" , search_kwargs ={"k" : 4})


prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


output_parser = StrOutputParser()

question = "What is this video mainly about?"

docs = retriver.invoke(question)
context = format_docs(docs)

final_prompt = prompt.format(
    context=context,
    question=question
)

chain = model | output_parser

answer = chain.invoke(final_prompt)

print(answer)