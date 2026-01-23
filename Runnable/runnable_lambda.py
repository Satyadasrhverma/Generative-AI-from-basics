from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# Prompts
prompt_notes = PromptTemplate(
    template="Write 3 short notes about {topic}",
    input_variables=["text"]
)

prompt_quiz = PromptTemplate(
    template="Create 2 question answers from {text}",
    input_variables=["text"]
)

# Parallel chain
parallel_chain = RunnableParallel({
    "notes": prompt_notes | model | parser,
    "quiz": prompt_quiz | model | parser,
    "short_text": RunnableLambda(lambda x: x["text"][:50])
})

# Run
result = parallel_chain.invoke({
    "topic" : "Python",
    "text": "Python is a popular programming language used in AI and web development."
})

print(result)
