from typing import Annotated
from typing_extensions import TypedDict
from llama_tool_wrapper import MyLlamaCppWithTools, improved_call_tools, improved_should_continue
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from tools_definition import tools
from dotenv import load_dotenv
import re

load_dotenv()

# Initialize the local LLaMA model
llm = MyLlamaCppWithTools("models/llama-3.1-8b-instruct-q4_k_m.gguf")

# Bind the model to available tools so it can perform tool calls
llm_with_tools = llm.bind_tools(tools)

# Define the state structure for the chatbot
# This state will track all messages in the conversation
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    """
    Generate a response using the LLM with tool support.

    Args:
        state (State): The current conversation state, containing messages.

    Returns:
        dict: Updated state containing the assistant's message.
    """
    try:
        print(f"Chatbot processing: {len(state['messages'])} messages")
        # Invoke the model with the current conversation history
        response = llm_with_tools.invoke(state["messages"])
        print(f"LLM response: {response.content[:100]}...")  # Preview first 100 characters
        return {"messages": [response]}
    except Exception as e:
        # Catch errors and return a fallback response
        print(f"Chatbot error: {e}")
        import traceback
        traceback.print_exc()
        return {"messages": [AIMessage(content="I encountered an error. Please try again")]}

def call_tools(state: State):
    """
    Execute any tool calls found in the last message.

    Args:
        state (State): The current conversation state.

    Returns:
        dict: Updated state after tool execution.
    """
    return improved_call_tools(state, llm_with_tools, tools)

def should_continue(state: State) -> str:
    """
    Determine if the conversation should continue with a tool call or end.

    Args:
        state (State): The current conversation state.

    Returns:
        str: 'tools' if tools need to be called, 'end' to finish conversation.
    """
    return improved_should_continue(state, llm_with_tools)

# Build the state graph for managing conversation flow
graph_builder = StateGraph(State)

# Add nodes: chatbot node generates responses, tools node executes tool calls
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", call_tools)

# Start the conversation with the chatbot node
graph_builder.add_edge(START, "chatbot")

# Conditional edges: after the chatbot responds, decide whether to call tools or end
graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# After tool execution, return to the chatbot to generate a response
graph_builder.add_edge("tools", "chatbot")

# Compile the graph for execution
graph = graph_builder.compile()

def main():
    """
    Main loop for interacting with the chatbot via command line.
    Users can type messages and receive responses.
    """
    print("JSON Tool-Calling Chatbot initialized. Type 'quit' to exit")
    
    while True:
        user_input = input("\nEnter a message: ").strip()
        
        # Exit commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_input:
            # Ignore empty messages
            print("Please enter a message")
            continue
            
        try:
            print(f"Processing: {user_input}")
            
            # Invoke the state graph with the new user message
            result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
            
            print(f"Graph result: {len(result['messages'])} messages")
            
            if result["messages"]:
                # --- Simplified Final Message Handling ---
                final_response = result["messages"][-1]

                if isinstance(final_response, AIMessage):
                    content = final_response.content
                    
                    # Remove JSON tool call syntax for clean output
                    cleaned_content = re.sub(r'```json.*?```', '', content, flags=re.DOTALL).strip()
                    cleaned_content = re.sub(r'"tool_calls":\s*\[.*?\]', '', cleaned_content, flags=re.DOTALL).strip()
                    cleaned_content = re.sub(r'[{}]', '', cleaned_content).strip()
                    
                    final_text = cleaned_content if cleaned_content else content
                    print(f"\nAssistant: {final_text}")
                else:
                    # Fallback for cases where the graph ends on a non-AI message
                    print(f"\nAssistant ended with a non-standard message: {final_response.content if hasattr(final_response, 'content') else str(final_response)}")
            else:
                print("No response received")
                
        except Exception as e:
            # Catch any unexpected errors during processing
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("Please try again")

# Entry point for the script
if __name__ == "__main__":
    main()