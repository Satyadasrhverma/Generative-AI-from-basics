from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
) 

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template= 'give me the name,age and city of a fictional person\n {format_instruction}',
    partial_variables= {'format_instruction': parser.get_format_instructions()}
    
)

#prompt = template.format()
#result = model.invoke(prompt)

#out =parser.parse(result.content)
#print(out)

chain = template | model | parser

result = chain.invoke({})

print(result)