from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

loader = TextLoader('RAG Components/Text Loader/rag_learning_notes.txt' , encoding= 'utf-8')
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


prompt = PromptTemplate(
    template='write a 5 mcq with answer from the following{text}',
    input_variables=['text']

)

parser = StrOutputParser()

docs = loader.load()

chain = prompt | model | parser 

print(chain.invoke({'text' : docs[0].page_content}))