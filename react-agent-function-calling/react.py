from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
load_dotenv()

@tool
def triple(num:float) -> float:
    """
    param num: a number to triple
    returns: the input multiplied by 3
    """
    return num * 3

tools = [TavilySearch(max_results=1), triple]    

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)