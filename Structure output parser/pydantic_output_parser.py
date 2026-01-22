from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# LLM setup
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name : str = Field(description="Name of the person")
    age : int = Field(gt= 18, description="Age of the person")
    city : str = Field(description= "Name of the city person belogs to")

parser = PydanticOutputParser(pydantic_object= Person)

template = PromptTemplate(
    template= 'Generate the name , age and city of a fictional {person} person\n {format_instruction}',
    input_variables= ['person'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'person': 'Indian'})

print(result)