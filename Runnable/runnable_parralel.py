from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

# LLM setup
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template= "Generate a tweet about {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= "Generate a linkedin post about {topic}",
    input_variables= ['topic']
)

parser = StrOutputParser()

parrell_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1 | model | parser),
    'linkedin' : RunnableSequence(prompt2 | model | parser)

})

result = parrell_chain.invoke({'topic' : 'Computer'})
print(result)