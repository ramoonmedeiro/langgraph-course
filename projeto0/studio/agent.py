from typing import TypedDict
from langgraph.graph import START, END, StateGraph
import random
from typing import Literal

class State(TypedDict):
    mensagem_humor: str

# Criando função de decisão aleatória
def tomador_de_decisao(state: State) -> Literal["no_2", "no_3"]:
    if random.random() < 0.5 :
        return "no_2"
    return "no_3"

def no_1(state: State):
    print("Entrou no nó 1")
    return {
        "mensagem_humor": state["mensagem_humor"] + " Eu estou"
    }

def no_2(state: State):
    print("Entrou no nó 2")
    return {
        "mensagem_humor": state["mensagem_humor"] + " tristinho."
    }

def no_3(state: State):
    print("Entrou no nó 3")
    return {
        "mensagem_humor": state["mensagem_humor"] + " felizão."
    }


builder = StateGraph(State)

# Adicionando nós
builder.add_node("no_1", no_1)
builder.add_node("no_2", no_2)
builder.add_node("no_3", no_3)

# Colocando a conexão dos nós

builder.add_edge(START, "no_1")
builder.add_conditional_edges("no_1", tomador_de_decisao)
builder.add_edge("no_2", END)
builder.add_edge("no_3", END)

# Compilando o grafo
graph = builder.compile()