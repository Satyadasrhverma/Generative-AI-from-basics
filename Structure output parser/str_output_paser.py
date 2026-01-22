from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
) 

model = ChatHuggingFace(llm = llm)


template1 = PromptTemplate(
    template= 'write a detailed report on {topic}',
    input_variables= ['topic']
    
)

template2 = PromptTemplate(
    template= "write a 5 point of line summary on the following text./n {text}",
    input_variables= ['text']
)

prompt = template1.invoke({'topic' : "Phsyics"})

result = model.invoke(prompt)

prompt2 = template2.invoke({'text' : result.content})

out = model.invoke(prompt2)
print(out.content)