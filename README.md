# LangGraph Llama.cpp Starter

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A powerful and easy-to-use starter template for building local, private, tool-calling LLM agents with LangGraph and `llama-cpp-python`.

This repository provides all the necessary code to run a sophisticated agent that can reason, use tools (like a calculator), and hold a conversation—all running on your own machine, no API keys required.

## Key Features

* **Local & Private**: Run a powerful AI agent completely offline. Your data never leaves your machine.
* **Tool-Calling Ready**: Built from the ground up to support JSON-based tool calling with local models.
* **Efficient Inference**: Uses `llama-cpp-python` for fast and efficient CPU/GPU inference with GGUF models.
* **Stateful Agents with LangGraph**: Implements a robust, stateful agent graph that can intelligently decide when to call tools and when to respond to the user.
* **Easy to Customize**: Add your own custom tools by following the simple template in `tools_definition.py`.

## Getting Started

Follow these steps to get the agent up and running.

### Prerequisites

* Python 3.10 or higher
* Git
* C++ build tools (required for `llama-cpp-python`). Refer to the [official installation guide](https://github.com/abetlen/llama-cpp-python#installation) for your specific OS.

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
```

### 2. Set Up a Virtual Environment

It's highly recommended to use a virtual environment.

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages. *Note: The `llama-cpp-python` installation may take a few minutes as it needs to be compiled.*

```bash
pip install -r requirements.txt
```

### 4. Download an LLM Model

This project is optimized for instruction-tuned GGUF models. We recommend a Llama 3.1 model.

1.  Download a model like the **[Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)**.
2.  Create a `models` folder in the root of this project.
3.  Place the downloaded `.gguf` file inside the `models` folder.

The code expects the model to be at `models/llama-3.1-8b-instruct-q4_k_m.gguf`. If you use a different model, remember to update the path in `main.py`.

## Usage

Once the setup is complete, you can start the chatbot.

```bash
python main.py
```

The script will load the model and you'll be prompted to enter a message.

**Example Interaction:**
```
JSON Tool-Calling Chatbot initialized. Type 'quit' to exit

Enter a message: What is 4 multiplied by 18?
Processing: What is 4 multiplied by 18?
Chatbot processing: 1 messages
LLM response: ```json
{
"tool_calls": [
{
"name": "multiply_numbers",
"arguments": {"a": 4, "...
Checking for tool calls in: ```json
{
"tool_calls": [
{
"name": "multiply_numbers",
"arguments": {"a": 4, ...
Found 1 tool calls, routing to: tools
Processing content for JSON tool calls: ```json ...
Parsed tool calls: [{'name': 'multiply_numbers', 'arguments': {'a': 4, 'b': 18}}]
Tool results: [ToolMessage(content='72.0', tool_call_id='multiply_numbers_0')]
Chatbot processing: 3 messages
LLM response: 4 multiplied by 18 is 72.
...

Assistant: 4 multiplied by 18 is 72.
```

## How to Customize

### Adding New Tools

Adding your own custom tools is easy:

1.  Open the `tools_definition.py` file.
2.  Follow the commented-out **template** at the top of the file to define your tool's arguments (using Pydantic) and the tool's function.
3.  Add your new tool function to the `tools` list at the bottom of the file.

### Changing the Model

To use a different model, simply update the model path string in `main.py`:
```python
# In main.py
llm = MyLlamaCppWithTools("models/YOUR-NEW-MODEL-NAME.gguf")
```

## How It Works

This project uses a few key components to function:

* **`main.py`**: The main entry point. It initializes the model, defines the LangGraph state and nodes, and runs the command-line interface.
* **`llama_tool_wrapper.py`**: The core logic. This custom wrapper class formats the system prompt with tool definitions and parses the model's JSON output to detect tool calls.
* **`tools_definition.py`**: A library of all available tools that the agent can use. This is where you'll add your own custom functions.
* **`llm_wrapper.py`**: A base wrapper for the `llama-cpp-python` library, handling the low-level model invocation.
