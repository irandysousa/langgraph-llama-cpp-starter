Perfect! Here’s a polished, professional version of your README without emojis:

---

# LangGraph Llama.cpp Starter

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A starter template for building local, private, tool-calling LLM agents using LangGraph and `llama-cpp-python`. This repository provides all the code needed to run an intelligent agent that can reason, call tools, and hold conversations—fully offline and under your control.

---

## Quickstart

Run the chatbot in 5 simple steps:

1. **Clone the repository**

```bash
git clone https://github.com/Dhyanesh18/langgraph-llama-cpp-starter.git
cd langgraph-llama-cpp-starter
```

2. **Set up a virtual environment**

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download a model**
   Place a GGUF LLaMA model (e.g., `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`) in the `models` folder.

5. **Start the chatbot**

```bash
python main.py
```

Then type your messages and the agent will respond, using tools automatically if needed.
For full setup, customization, and troubleshooting, see the sections below.


## Key Features

* **Local & Private**: Run AI agents entirely on your machine. No data is sent to external servers.
* **Tool-Calling Ready**: Supports JSON-based tool calling, allowing agents to interact with calculators, APIs, or custom functions.
* **Efficient Inference**: Uses `llama-cpp-python` for fast CPU/GPU inference with GGUF models, including 4-bit/8-bit quantized models.
* **Stateful Agents**: Implements a LangGraph state machine to manage multi-step reasoning, deciding when to call tools and when to respond directly.
* **Easy Customization**: Add new tools and modify agent behavior by editing simple Python files.

## Why LangGraph + LLaMA-Cpp?

Combining LangGraph with `llama-cpp-python` allows you to create agents that:

* Maintain conversation state across multiple turns.
* Perform structured reasoning with conditional tool calls.
* Run completely offline, unlike cloud solutions like Ollama.

---

## Getting Started

### Prerequisites

* Python 3.10 or higher
* Git
* C++ build tools (required for `llama-cpp-python`)
  See the [official installation guide](https://github.com/abetlen/llama-cpp-python#installation) for your OS.

### 1. Clone the Repository

```bash
git clone https://github.com/Dhyanesh18/langgraph-llama-cpp-starter.git
cd langgraph-llama-cpp-starter
```

### 2. Set Up a Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download an LLM Model

This project is optimized for instruction-tuned GGUF models (e.g., LLaMA 3.1).

1. Download a model like [Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf).
2. Create a `models` folder in the project root.
3. Place the `.gguf` file inside `models`.
4. Update the model path in `main.py` if using a different filename.

---

## Usage

Start the chatbot:

```bash
python main.py
```

Example interaction:

```
Enter a message: What is 4 multiplied by 18?
Assistant: 4 multiplied by 18 is 72.
```

The agent can detect and execute tool calls automatically, then provide a final natural language response.

---

## Customization

### Adding New Tools

1. Open `tools_definition.py`.
2. Follow the template to define tool arguments and functions.
3. Add your new tool to the `tools` list.

### Changing the Model

Update the model path in `main.py`:

```python
llm = MyLlamaCppWithTools("models/YOUR-MODEL-NAME.gguf")
```

---

## How It Works

* **`main.py`**: Initializes the model, defines the LangGraph state, and runs the CLI.
* **`llama_tool_wrapper.py`**: Wraps the LLaMA model to handle tool calls and parse JSON responses.
* **`tools_definition.py`**: Library of all available tools. Add custom functions here.
* **`llm_wrapper.py`**: Base wrapper for `llama-cpp-python` model invocation.

---

## Troubleshooting

* **Memory Issues**: Adjust `n_blocks` in `main.py` according to your GPU’s dedicated memory.
* **Attribute Errors**: Ensure your `llama_tool_wrapper.py` version matches the LLaMA wrapper you are using.
* **Tool Misbehavior**: Check your JSON tool format follows the required schema.

---

## Contributing

PRs and suggestions are welcome!
--
