import json
import typing

import requests

from agent.llm.deepseek.exception_handler import deepseek_exception
from agent.llm.role import Role

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_FLASH = "deepseek-flash"
DEFAULT_SYSTEM_PROMPT="You are a helpful assistant"
DEFAULT_MAX_TOKENS = 4096

class DeepseekClient:
    def __init__(self, api_key: str,
                 model_name: str=DEEPSEEK_FLASH,
                 system_prompt: str=DEFAULT_SYSTEM_PROMPT,
                 max_tokens: int=DEFAULT_MAX_TOKENS):
        self.api_key: str= api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

        self.history: list = [{
            'role': Role.SYSTEM,
            'content': self.system_prompt,
        }]

    def _headers(self, stream: bool = False):
        return {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream' if stream else 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def _build_payload(self, target_messages: list[dict], stream: bool = False) -> dict:
        return {
            "messages": target_messages,
            "model": self.model_name,
            "thinking": {
                "type": "disabled",
            },
            "reasoning_effort": "none",
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "text"
            },
            "stop": None,
            "stream": stream,
            "stream_options": None,
            "temperature": 1,
            "top_p": 1,
            "tools": None,
            "tool_choice": "none",
            "logprobs": False,
            "top_logprobs": None,
        }

    def chat(self, human_message: str, thinking: str = "disabled", reasoning_effort: str = "none") -> str:
        target_messages = self.history.copy()
        target_messages.append({'role': Role.USER, 'content': human_message})
        payload = self._build_payload(target_messages, stream=False)
        headers = self._headers(stream=False)
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            json_response = json.loads(response.text)
            llm_response_text = json_response['choices'][0]['message']['content']
            self.history.append({'role': Role.USER, 'content': human_message})
            self.history.append({'role': Role.ASSISTANT, 'content': llm_response_text})
            return llm_response_text
        raise deepseek_exception(response.status_code, response.text)

    def stream(self, human_message: str, thinking: str = "disabled", reasoning_effort: str = "none") -> \
    typing.Generator[str, None, None]:
        target_messages = self.history.copy()
        target_messages.append({'role': Role.USER, 'content': human_message})
        payload = self._build_payload(target_messages, stream=True)
        headers = self._headers(stream=True)
        with requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, stream=True, timeout=30) as response:
            if response.status_code != 200:
                raise deepseek_exception(response.status_code, response.text)
            full_content = []
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                content = chunk["choices"][0]["delta"].get("content")
                if content:
                    full_content.append(content)
                    yield content
        complete_response = "".join(full_content)
        self.history.append({'role': Role.USER, 'content': human_message})
        self.history.append({'role': Role.ASSISTANT, 'content': complete_response})