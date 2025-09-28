from llm_wrapper import MyLlamaCpp
import json
import re
from typing import Dict, Any, List
from langchain_core.messages import AIMessage, ToolMessage

class MyLlamaCppWithTools(MyLlamaCpp):
    """LLM wrapper that adds JSON-based tool calling support."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the LLM and prepare tool bindings."""
        super().__init__(*args, **kwargs)
        self._bound_tools = []

    def bind_tools(self, tools):
        """
        Bind a list of tools to this LLM instance.

        Args:
            tools (list): List of tool objects with 'name', 'description', and callable 'func'.

        Returns:
            self: Returns the current instance for method chaining.
        """
        self._bound_tools = tools
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
            system_msg = f"""You are a helpful assistant with access to tools. When you need to use a tool, respond with a JSON object in this exact format:
            ```json
            {{
            "tool_calls": [
                {{
                "name": "tool_name",
                "arguments": {{"param1": "value1", "param2": "value2"}}
                }}
            ]
            }}
            ```
            Available tools:
            {self._get_tool_schema()}

            Important rules:
            1. Only use tools when necessary
            2. Use proper JSON formatting
            3. Include all required parameters
            4. You can call multiple tools in one response
            5. After tool results, provide a natural language response

            For simple questions that don't require tools, respond normally without JSON.
            """

            prompt_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>")
        
        # Add conversation messages
        for message in messages:
            # Handle ToolMessages with the correct Llama 3 format 
            if isinstance(message, ToolMessage):
                prompt_parts.append(f"<|start_header_id|>tool<|end_header_id|>\n\n{message.content}<|eot_id|>")
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
        Extract JSON tool calls from the LLM response.

        Args:
            response_content (str): The text response from the LLM.

        Returns:
            List[Dict]: List of parsed tool call dictionaries.
        """
        tool_calls = []
        
        # Look for JSON blocks in the response
        json_pattern = r'```json\s*(.*?)\s*```'
        json_matches = re.findall(json_pattern, response_content, re.DOTALL)
        
        for json_text in json_matches:
            try:
                parsed = json.loads(json_text)
                if "tool_calls" in parsed:
                    tool_calls.extend(parsed["tool_calls"])
            except json.JSONDecodeError:
                continue
        
        # Also try to find JSON without code blocks
        if not tool_calls:
            try:
                # Try to parse the entire response as JSON
                parsed = json.loads(response_content.strip())
                if "tool_calls" in parsed:
                    tool_calls.extend(parsed["tool_calls"])
            except json.JSONDecodeError:
                pass
        
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
            
            # Find the tool
            tool_function = None
            for tool in available_tools:
                if tool.name == tool_name:
                    tool_function = tool
                    break
            
            if tool_function:
                try:
                    # Call with keyword arguments
                    result = tool_function.invoke(arguments)
                    results.append(ToolMessage(
                        content=str(result), 
                        tool_call_id=f"{tool_name}_{i}"
                    ))
                except Exception as e:
                    results.append(ToolMessage(
                        content=f"Error calling {tool_name}: {str(e)}", 
                        tool_call_id=f"{tool_name}_{i}"
                    ))
            else:
                results.append(ToolMessage(
                    content=f"Unknown tool: {tool_name}", 
                    tool_call_id=f"unknown_{i}"
                ))
        
        return results


# Standalone functions for use with graph
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
    
    print(f"Processing content for JSON tool calls: {content}")
    
    # Parse tool calls
    tool_calls = llm_with_tools.parse_tool_calls(content)
    print(f"Parsed tool calls: {tool_calls}")
    
    if not tool_calls:
        print("No tool calls found")
        return {"messages": []}
    
    # Execute tool calls
    tool_results = llm_with_tools.execute_tool_calls(tool_calls, tools)
    print(f"Tool results: {tool_results}")
    
    return {"messages": tool_results}


def improved_should_continue(state, llm_with_tools) -> str:
    """Check if response contains JSON tool calls."""
    last_message = state["messages"][-1]
    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    print(f"Checking for tool calls in: {content[:100]}...")
    
    # Check for JSON tool calls
    tool_calls = llm_with_tools.parse_tool_calls(content)
    result = "tools" if tool_calls else "end"
    
    print(f"Found {len(tool_calls)} tool calls, routing to: {result}")
    return result