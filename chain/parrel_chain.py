from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

# LLM setup
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


prompt1= PromptTemplate(
    template= 'Generate simple notes from the following text\n {text}',
    input_variables= ['text']
)

prompt2 = PromptTemplate(
    template= 'generate 5 short question answer from the following \n {topic}',
    input_variables= ['topic']
)

prompt3 = PromptTemplate(
    template= 'merge the provide notes and quiz into a single Document \n notes -> {notes} and quiz -> {quiz}',
    input_variables= ['notes', 'quiz']
)

parser = StrOutputParser()

parrallel_chain = RunnableParallel({
    'notes' : prompt1 | model | parser,
    'quiz' : prompt2 | model | parser

})

sample_text = """
Artificial Intelligence (AI) is a branch of computer science that focuses on
creating machines capable of performing tasks that typically require human
intelligence. These tasks include learning from data, recognizing patterns,
understanding natural language, and making decisions.

AI is widely used in applications such as virtual assistants, recommendation
systems, autonomous vehicles, healthcare diagnostics, and fraud detection.
With advancements in machine learning and deep learning, AI systems are
becoming more accurate and efficient.
"""

merge_chain = prompt3 | model | parser

chain = parrallel_chain | merge_chain

result = chain.invoke({
    "text": sample_text,
    "topic": "Artificial Intelligence"
})

print(result)
