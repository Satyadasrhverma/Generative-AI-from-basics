from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

url = 'https://leetcode.com/problemset/'
loader = WebBaseLoader(url)
docs = loader.load()


# 3️⃣ Load LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template= 'Answer the following {Question}\n form the followinhg -\n {text}',
    input_variables= ['Question' , 'text']


)

parser = StrOutputParser()
chain = prompt| model | parser

print(chain.invoke({
    'Question' : 'what is this question',
    'text' : docs[0].page_content
}))
