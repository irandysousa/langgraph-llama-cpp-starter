from llm_wrapper import MyLlamaCpp
import json
import re
from typing import Dict, Any, List
from langchain_core.messages import AIMessage, ToolMessage
import logging

logger = logging.getLogger(__name__)

class MyLlamaCppWithTools(MyLlamaCpp):
    """LLM wrapper that adds JSON-based tool calling support."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the LLM and prepare tool bindings."""
        super().__init__(*args, **kwargs)
        self._bound_tools = []
        self._use_native_tool_role = kwargs.get('use_native_tool_role', True)
    
    def bind_tools(self, tools):
        """
        Bind a list of tools to this LLM instance.
        Args:
            tools (list): List of tool objects with 'name', 'description', and callable 'func'.
        Returns:
            self: Returns the current instance for method chaining.
        """
        self._bound_tools = tools
        logger.info(f"Bound {len(tools)} tools to LLM")
        return self
    
    def invoke(self, messages):
        """
        Generate a response from the LLM, including tool information if needed.
        Args:
            messages (list or str): Conversation messages.
        Returns:
            AIMessage: The LLM-generated response.
        """
        if isinstance(messages, list):
            prompt = self._messages_to_prompt_with_tools(messages)
        else:
            prompt = str(messages)
        
        response = self._call(prompt)
        return AIMessage(content=response)
    
    def _get_tool_schema(self) -> str:
        """
        Generate a JSON schema describing all bound tools from their Pydantic models.
        Returns:
            str: JSON-formatted schema of tools.
        """
        if not self._bound_tools:
            return ""
        
        tools_info = []
        for tool in self._bound_tools:
            if not hasattr(tool, 'args_schema') or not tool.args_schema:
                continue
            
            schema = tool.args_schema.schema()
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", [])
                }
            })
        
        return json.dumps(tools_info, indent=2)
    
    def _extract_role_content(self, message):
        """
        Extract role and content from various message formats.
        Args:
            message: A message object or dictionary.
        Returns:
            tuple: (role: str, content: str)
        """
        if hasattr(message, 'role') and hasattr(message, 'content'):
            role = message['role'] if isinstance(message, dict) else message.role
            content = message['content'] if isinstance(message, dict) else message.content
        elif isinstance(message, dict):
            role = message.get('role', 'user')
            content = message.get('content', '')
        else:
            if hasattr(message, 'type'):
                role = 'assistant' if message.type == 'ai' else 'user'
            else:
                role = 'user'
            content = str(message.content) if hasattr(message, 'content') else str(message)
        return role, content
    
    def _messages_to_prompt_with_tools(self, messages):
        """Convert messages to prompt with tool information using JSON format."""
        prompt_parts = []
        
        if self._bound_tools:
            system_msg = f"""You are a helpful, smart, and efficient assistant with access to tools.
            Your task is to answer the user's request by breaking it down into a series of logical steps. For each step, you must decide whether to use a tool or to respond directly to the user.

            **TOOL USAGE RULES:**
            1.  **One Tool at a Time**: You MUST call only ONE tool at a time. For multi-step tasks, you will call a tool, receive the result, and then decide the next step in a new turn.
            2.  **DO NOT MAKE UP TOOLS WHICH ARE NOT GIVEN BELOW**
            3.  **JSON ONLY for Tool Calls**: When you decide to call a tool, your response MUST be ONLY a single JSON object enclosed in ```json code blocks. Nothing else. No explanations, no conversation, no introductory text.
                **Correct Format:**
                ```json
                {{
                "tool_calls": [
                    {{
                    "name": "tool_name",
                    "arguments": {{"arg1": "value1"}}
                    }}
                ]
                }}
                ```
            4.  **Trust Tool Results**: When you receive a tool result, trust its content. Use that result to determine the next logical step. Do not re-run a step that has already been completed successfully.
            5.  **Final Answer**: Once all necessary tool calls are complete and you have the final answer, respond to the user in natural language without any JSON.
            6.  **No Tools Needed**: If the user's request cannot be answered using the available tools, you may respond directly in natural language without attempting any tool call
            7.  **Avoid Irrelevant Tools**: Only call a tool if it clearly applies to the current step. Do not make dummy calculations or irrelevant calls.
            
            **Available tools:**
            {self._get_tool_schema()}
            """
            prompt_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>")
        
        for message in messages:
            if isinstance(message, ToolMessage):
                if self._use_native_tool_role:
                    # Use native tool role (Llama 3.1+)
                    prompt_parts.append(f"<|start_header_id|>tool<|end_header_id|>\n\n{message.content}<|eot_id|>")
                else:
                    # Fallback for models without tool role training
                    prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\nTool Result: {message.content}<|eot_id|>")
                continue
            
            role, content = self._extract_role_content(message)
            if role == 'user':
                prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == 'assistant':
                prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
        
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(prompt_parts)
    
    def parse_tool_calls(self, response_content: str) -> List[Dict[str, Any]]:
        """
        Extract JSON tool calls from the LLM response with improved error handling.
        """
        tool_calls = []
        
        # Try code block first (```json ... ```)
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, response_content, re.DOTALL | re.IGNORECASE)
        
        if match:
            json_text = match.group(1).strip()
        else:
            # Try to find raw JSON object with tool_calls
            json_pattern = r'\{[^{}]*"tool_calls"[^{}]*\[[^\]]*\][^{}]*\}'
            match = re.search(json_pattern, response_content, re.DOTALL)
            if match:
                json_text = match.group(0)
            else:
                logger.debug("No JSON tool calls found in response")
                return tool_calls
        
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                logger.info(f"Found {len(parsed['tool_calls'])} tool call(s)")
                return parsed["tool_calls"]
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            logger.debug(f"Failed to parse: {json_text[:200]}")
        
        return tool_calls
    
    def execute_tool_calls(self, tool_calls: List[Dict], available_tools: List) -> List[ToolMessage]:
        """
        Execute a list of parsed tool calls.
        Args:
            tool_calls (List[Dict]): List of tool call dictionaries.
            available_tools (List): List of bound tool objects.
        Returns:
            List[ToolMessage]: List of messages containing tool outputs.
        """
        results = []
        
        for i, call in enumerate(tool_calls):
            tool_name = call.get("name")
            arguments = call.get("arguments", {})
            
            logger.info(f"Executing tool: {tool_name} with args: {arguments}")
            
            tool_function = None
            for tool in available_tools:
                if tool.name == tool_name:
                    tool_function = tool
                    break
            
            if tool_function:
                try:
                    result = tool_function.invoke(arguments)
                    logger.info(f"Tool {tool_name} returned: {result}")
                    results.append(ToolMessage(
                        content=str(result), 
                        tool_call_id=f"{tool_name}_{i}"
                    ))
                except Exception as e:
                    error_msg = f"Error calling {tool_name}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    results.append(ToolMessage(
                        content=error_msg, 
                        tool_call_id=f"{tool_name}_{i}"
                    ))
            else:
                error_msg = f"Unknown tool: {tool_name}"
                logger.warning(error_msg)
                results.append(ToolMessage(
                    content=error_msg, 
                    tool_call_id=f"unknown_{i}"
                ))
        
        return results


def improved_call_tools(state, llm_with_tools, tools):
    """
    Parse and execute tool calls from the last message in the state.
    Args:
        state (dict): Current conversation state.
        llm_with_tools (MyLlamaCppWithTools): LLM instance.
        tools (list): List of available tools.
    Returns:
        dict: Updated state with tool execution messages.
    """
    last_message = state["messages"][-1]
    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    logger.debug(f"Processing content for JSON tool calls: {content[:200]}")
    
    tool_calls = llm_with_tools.parse_tool_calls(content)
    
    if not tool_calls:
        logger.debug("No tool calls found")
        return {"messages": []}
    
    tool_results = llm_with_tools.execute_tool_calls(tool_calls, tools)
    logger.info(f"Executed {len(tool_results)} tool(s)")
    
    return {"messages": tool_results}


def improved_should_continue(state, llm_with_tools, max_iterations: int = 20) -> str:
    """
    Check if response contains JSON tool calls and prevent infinite loops.
    Args:
        state (dict): Current conversation state.
        llm_with_tools (MyLlamaCppWithTools): LLM instance.
        max_iterations (int): Maximum number of conversation turns before forcing end.
    Returns:
        str: 'tools' if tools need to be called, 'end' to finish conversation.
    """
    messages = state["messages"]
    
    # Check for infinite loop prevention
    if len(messages) > max_iterations:
        logger.warning(f"Maximum iterations ({max_iterations}) reached. Ending conversation.")
        return "end"
    
    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    logger.debug(f"Checking for tool calls in: {content[:100]}...")
    
    tool_calls = llm_with_tools.parse_tool_calls(content)
    result = "tools" if tool_calls else "end"
    
    logger.info(f"Found {len(tool_calls)} tool calls, routing to: {result}")
    return result