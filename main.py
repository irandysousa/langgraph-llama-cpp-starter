from typing import Annotated
from typing_extensions import TypedDict
from llama_tool_wrapper import MyLlamaCppWithTools, improved_call_tools, improved_should_continue
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from tools_definition import tools
from dotenv import load_dotenv
import os
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Get model path from environment variable with fallback
MODEL_PATH = os.getenv("MODEL_PATH", "models/llama-3.1-8b-instruct-q4_k_m.gguf")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "20"))
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "false").lower() == "true"

# Initialize the local LLaMA model
logger.info(f"Initializing LLM with model: {MODEL_PATH}")
llm = MyLlamaCppWithTools(MODEL_PATH)
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    """State structure for the chatbot with message history."""
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
        logger.debug(f"Chatbot processing: {len(state['messages'])} messages")
        response = llm_with_tools.invoke(state["messages"])
        logger.debug(f"LLM response: {response.content[:100]}...")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Chatbot error: {e}", exc_info=True)
        return {"messages": [AIMessage(content="I encountered an error. Please try again.")]}


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
    return improved_should_continue(state, llm_with_tools, max_iterations=MAX_ITERATIONS)


# Build the state graph for managing conversation flow
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", call_tools)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()


def clean_response(content: str) -> str:
    """
    Remove JSON tool call syntax from final response for clean output.
    Args:
        content (str): Raw response content.
    Returns:
        str: Cleaned response content.
    """
    # Remove JSON code blocks
    cleaned = re.sub(r'```json.*?```', '', content, flags=re.DOTALL).strip()
    # Remove tool_calls JSON structure
    cleaned = re.sub(r'"tool_calls":\s*\[.*?\]', '', cleaned, flags=re.DOTALL).strip()
    # Remove stray braces
    cleaned = re.sub(r'^\s*[{}]\s*$', '', cleaned, flags=re.MULTILINE).strip()
    
    return cleaned if cleaned else content


def main():
    """
    Main loop for interacting with the chatbot via command line.
    Users can type messages and receive responses.
    """
    logger.info("JSON Tool-Calling Chatbot initialized")
    print("\n" + "="*60)
    print("JSON Tool-Calling Chatbot")
    print("="*60)
    print("Commands:")
    print("  - Type your message to chat")
    print("  - 'quit', 'exit', or 'q' to exit")
    print("  - 'clear' to clear conversation history")
    print("="*60 + "\n")
    
    conversation_state = {"messages": []}
    
    while True:
        user_input = input("\nYou: ").strip()
        
        # Exit commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        # Clear conversation history
        if user_input.lower() == 'clear':
            conversation_state = {"messages": []}
            logger.info("Conversation history cleared")
            print("Conversation history cleared")
            continue
            
        if not user_input:
            print("Please enter a message")
            continue
            
        try:
            logger.info(f"Processing user input: {user_input}")
            
            # Add user message to conversation state
            conversation_state["messages"].append(HumanMessage(content=user_input))
            
            # Invoke the state graph
            result = graph.invoke(conversation_state)
            
            # Update conversation state with results
            conversation_state = result
            
            logger.debug(f"Graph result: {len(result['messages'])} messages")
            
            if result["messages"]:
                final_response = result["messages"][-1]
                
                if isinstance(final_response, AIMessage):
                    content = final_response.content
                    final_text = clean_response(content)
                    print(f"\nAssistant: {final_text}")
                else:
                    # Handle edge case where final message is not AIMessage
                    response_content = final_response.content if hasattr(final_response, 'content') else str(final_response)
                    print(f"\nAssistant: {response_content}")
            else:
                logger.warning("No response received from graph")
                print("\nNo response received")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error during conversation: {e}", exc_info=True)
            print(f"\nError: {e}")
            print("Please try again or type 'quit' to exit")


if __name__ == "__main__":
    main()