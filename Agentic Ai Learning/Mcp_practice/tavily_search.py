import os 
from dotenv import load_dotenv
load_dotenv()
from mcp.server.mcpserver import MCPServer
from tavily import TavilyClient

# Initialize Tavily tavily_client = TavilyClient(api_key=api_key)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

mcp = MCPServer("TavilySearch")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def tavily_search(query:str,max_results:int=2) ->str:
    """Search for information using Tavily."""
    results = tavily_client.search(query, max_results=max_results)
    
    results = results.get("results", []) 
    if not results: 
        return "No search results found." 
    formatted_results = [] 
    for index, result in enumerate(results, start=1): 
        formatted_results.append(
            f"Result {index}\n"
            f"Title: {result.get('title', 'N/A')}\n"
            f"URL: {result.get('url', 'N/A')}\n" f"Content: {result.get('content', 'N/A')}\n"
            ) 
    return "\n".join(formatted_results)



if __name__ == "__main__":
    mcp.run(transport="stdio")