from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from typing import List
from pydantic import BaseModel, Field
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END


from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

serper_api_key = os.getenv("SERPER_API_KEY")

class GetAllLinksInput(BaseModel):
    query: str = Field(description="Query que a llm deverá trazer os links associados")

@tool(name_or_callable="get-all-links", args_schema=GetAllLinksInput)
def get_all_links(query: str) -> List[str]: 
    """
    Realiza uma busca na web usando a Google Serper API e extrai uma lista
    dos URLs orgânicos dos resultados.

    Esta função inicializa um wrapper para a Serper API com configurações
    específicas para busca no Brasil e em português, limitando os resultados
    aos 10 principais links orgânicos.

    Args:
        query (str): O termo ou frase de busca a ser usado na Serper API.
                     Exemplo: "notícias de inteligência artificial hoje".

    Returns:
        List[str]: Uma lista de strings, onde cada string é um URL
                   ("link") de um resultado de busca orgânico. A lista
                   conterá no máximo 10 links.

    """
    search_wrapper = GoogleSerperAPIWrapper(
        k=5,
        gl="br",
        hl="pt",
        serper_api_key=serper_api_key
    )   

    results = search_wrapper.results(query=query)
    links = [r["link"] for r in results["organic"]]
    return links


async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit(async_browser=async_browser)
playwright_tools = toolkit.get_tools()

tools_by_name = {tool.name: tool for tool in playwright_tools}
navigate_tool = tools_by_name["navigate_browser"]
click_element_tool = tools_by_name["click_element"]
extract_text_tool = tools_by_name["extract_text"]
all_tools = [get_all_links, navigate_tool, click_element_tool, extract_text_tool]

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
model_with_tools = model.bind_tools(all_tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Definir o nó do LLM
def chatbot(state: State):
    return {"messages": [model_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)

# Criar nó de ferramentas
tool_node = ToolNode(tools=all_tools)
builder.add_node("tools", tool_node)

# Se o modelo pedir uma ferramenta -> ir para "tools"
# Se não -> encerrar
builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Após usar a ferramenta, volta para o LLM
builder.add_edge("tools", "chatbot")

# Definir início
builder.set_entry_point("chatbot")

graph = builder.compile()
