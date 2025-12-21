from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from chains import reflection_chain, generation_chain




class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph) -> MessageGraph:
    return {"messages": generation_chain.invoke({"messages": state["messages"]})}

def reflection_node(state: MessageGraph) -> MessageGraph:
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph) -> str:
    """
    Check if the agent should continue.
    """
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue,{
    END:END,
    REFLECT:REFLECT
    })
builder.add_edge(REFLECT,GENERATE)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="flow.png")
graph.get_graph().print_ascii()

    


if __name__ == "__main__":
    inputs = {
        "messages": [HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
                                ]
                            }

    response = graph.invoke(inputs)
    print(response)                              
