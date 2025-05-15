
from pathlib import Path
import re
from litgpt.prompts import PromptStyle as LitGPTPromptStyle
from litgpt.config import Config


class PromptStyle(LitGPTPromptStyle):
    @classmethod
    def from_config(cls, config: Config) -> "PromptStyle":
        return model_name_to_prompt_style(config.name)
    

class Llama3(PromptStyle):

    ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def apply(self, messages):
        tokens = []
        for message in messages:
            if message["role"] not in ["system", "user", "assistant"]:
                raise ValueError(f"Invalid role: {message['role']}")
            message = (
                f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
                f"{message['content']}<|eot_id|>"  # No newline
            )
            tokens.append(message)
        return "".join(tokens)
    

class Pythia(PromptStyle):

    ASSISTANT_HEADER = "<|endoftext|>Assistant: "

    def apply(self, messages):
        tokens = []
        for message in messages:
            if message["role"] == "system":
                message = f"<|endoftext|>Instructions: {message['content']}\n"
            elif message["role"] == "user":
                message = f"<|endoftext|>User: {message['content']}\n"
            elif message["role"] == "assistant":
                message = f"<|endoftext|>Assistant: {message['content']}\n"
            else:
                raise ValueError(f"Invalid role: {message['role']}")
            tokens.append(message)
        return "".join(tokens)
        
        
def model_name_to_prompt_style(model_name):
    
    if re.search("Llama-3.*-Instruct", model_name):
        return Llama3()
    if re.search("Llama-3.*-Instruct-*", model_name):
        return Llama3()
    if re.search("pythia-*", model_name):
        return Pythia()