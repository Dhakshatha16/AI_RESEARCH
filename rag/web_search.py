from langchain_tavily import TavilySearch
import os

def create_web_search_tool():
    search_tool = TavilySearch(
        max_results=3,
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    print("✅ Web Search Tool created!")
    return search_tool