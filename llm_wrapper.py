from llama_cpp import Llama
from langchain_core.language_models import LLM
from pydantic import PrivateAttr
from langchain_core.messages import AIMessage
from typing import List, Optional

class MyLlamaCpp(LLM):
    _llama: Llama = PrivateAttr()

    def __init__(self, model_path, n_gpu_layers=28, n_threads=12, n_ctx=8192):
        super().__init__()
        self._llama = Llama(
            model_path = model_path,
            n_gpu_layers = n_gpu_layers,
            n_threads = n_threads,
            n_ctx = n_ctx
        )

    @property
    def _llm_type(self):
        return "llama-cpp-custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call the Llama model and return the response."""
        max_tokens = kwargs.get('max_tokens', 1024)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        repeat_penalty = kwargs.get('repeat_penalty', 1.1)
        
        try:
            output = self._llama(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=stop or ["<|eot_id|>", "<|end_of_text|>"],  # Add proper stop tokens
                **kwargs
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Error in LLM call: {e}")
            return "Sorry, I encountered an error processing your request."
        
    def invoke(self, messages):
        if isinstance(messages, list):
            prompt = self._messages_to_prompt(messages)
        else:
            prompt = str(messages)
        response = self._call(prompt)
        return AIMessage(content=response)

    def _messages_to_prompt(self, messages):
        """Convert a list of messages to a prompt string using Llama 3.1 chat format."""
        prompt_parts = []
        
        for message in messages:
            if hasattr(message, 'role') and hasattr(message, 'content'):
                # Handle dict-like messages
                role = message['role'] if isinstance(message, dict) else message.role
                content = message['content'] if isinstance(message, dict) else message.content
            elif isinstance(message, dict):
                role = message.get('role', 'user')
                content = message.get('content', '')
            else:
                # Handle BaseMessage objects
                if hasattr(message, 'type'):
                    role = 'assistant' if message.type == 'ai' else 'user'
                else:
                    role = 'user'
                content = str(message.content) if hasattr(message, 'content') else str(message)
            
            if role == 'user':
                prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == 'assistant':
                prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
        
        # Add the assistant header for the response
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(prompt_parts)