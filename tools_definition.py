from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------- 
#                TEMPLATE FOR ADDING A NEW CUSTOM TOOL                         
# ---------------------------------------------------------------------------- 
# To add your own tool, copy and modify the template below.                    
# ---------------------------------------------------------------------------- 

# 1. DEFINE THE ARGUMENTS SCHEMA
#    Create a Pydantic model for your tool's arguments. This ensures the
#    inputs are validated and provides clear descriptions for the LLM.
#
# class YourToolNameArgs(BaseModel):
#     argument_1: str = Field(description="A clear description of what this argument is for.")
#     argument_2: int = Field(description="A description for the second argument.")

# 2. CREATE THE TOOL FUNCTION
#    Use the `@tool` decorator and pass your arguments schema to it.
#    Write a clear docstring explaining what the tool does.
#
# @tool(args_schema=YourToolNameArgs)
# def your_tool_name(argument_1: str, argument_2: int) -> str:
#     """A clear, one-sentence description of what this tool does overall."""
#     # Your tool's logic goes here
#     result = f"You passed {argument_1} and {argument_2}"
#     return result

# 3. ADD YOUR NEW TOOL TO THE LIST
#    Finally, make sure to add your new tool function to the `tools` list
#    at the bottom of this file.
#
# ---------------------------------------------------------------------------- 

# Pydantic Schemas for tool arguments
class MathArgs(BaseModel):
    a: float = Field(description="The first floating point number.")
    b: float = Field(description="The second floating point number.")

class DivideArgs(BaseModel):
    a: float = Field(description="The dividend (the number being divided).")
    b: float = Field(description="The divisor (the number to divide by).")


@tool(args_schema=MathArgs)
def add_numbers(a: float, b: float) -> float:
    """Add two floating point numbers and return the result."""
    return a + b

@tool(args_schema=MathArgs)
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two floating point numbers and return the result."""
    return a * b

@tool(args_schema=MathArgs)
def subtract_numbers(a: float, b: float) -> float:
    """Subtract two floating point numbers and return the result."""
    return a - b

@tool(args_schema=DivideArgs)
def divide_numbers(a: float, b: float) -> float:
    """Divide two floating point numbers and return the result."""
    if b == 0:
        return "Error: Division by zero"
    return a / b

tools = [add_numbers, multiply_numbers, subtract_numbers, divide_numbers]