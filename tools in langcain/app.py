from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# -----------------------------
# 1️⃣ LLM SETUP (HuggingFace)
# -----------------------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# -----------------------------
# 2️⃣ TOOLS
# -----------------------------
@tool
def search(query: str) -> str:
    """Search information based on a query"""
    return f"🔍 Search result for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location"""
    return f"🌤 Weather in {location}: Sunny, 30°C"

tools = [search, get_weather]

# -----------------------------
# 3️⃣ SYSTEM PROMPT
# -----------------------------
system_prompt = """
You are a helpful assistant.

Rules:
- If the user asks to SEARCH something, use the search tool.
- If the user asks about WEATHER, use the get_weather tool.
- Do NOT guess information.
- Use tools when required.
"""

# -----------------------------
# 4️⃣ CREATE AGENT (STABLE)
# -----------------------------
agent = create_openai_tools_agent(
    llm=model,
    tools=tools,
    prompt=system_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# -----------------------------
# 5️⃣ RUN LOOP
# -----------------------------
while True:
    user_input = input("\nAsk something (search/weather) or 'exit': ")

    if user_input.lower() == "exit":
        print("👋 Exiting...")
        break

    response = agent_executor.invoke(
        {"input": user_input}
    )

    print("\n✅ FINAL ANSWER:")
    print(response["output"])
