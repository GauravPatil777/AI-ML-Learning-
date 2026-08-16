from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

llm=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# built in tool
duckduck_search=DuckDuckGoSearchRun()


# Custom tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a:int,b:int)->int:
    """Multiply two numbers"""
    return a*b

agent = create_agent(
    model=llm,
   tools=[multiply, add,duckduck_search ],
    system_prompt="You are a helpful assistant. Use tools when necessary."
)

inputs = {
    "messages": [
        ("user", "what is langchain and What is 25 multiplied by 8?")
    ]
}

result = agent.invoke(inputs)

print(result["messages"][-1].text)