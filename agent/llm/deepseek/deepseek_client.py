import json

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


    def _headers(self):
        return {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }


    def chat(self, human_message: str,
             thinking: str="disabled",
             reasoning_effort: str="none") -> str:
        target_messages: list[dict] = self.history.copy()
        target_messages.append({
            'role': Role.USER,
            'content': human_message
        })
        payload = json.dumps({
            "messages": target_messages,
            "model": self.model_name,
            "thinking": {
                "type": thinking,
            },
            "reasoning_effort": reasoning_effort,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "text"
            },
            "stop": None,
            "stream": False,
            "stream_options": None,
            "temperature": 1,
            "top_p": 1,
            "tools": None,
            "tool_choice": "none",
            "logprobs": False,
            "top_logprobs": None
        })
        headers = self._headers()

        response = requests.post(DEEPSEEK_API_URL, headers=headers, data=payload)
        if response.status_code == 200:
            json_response = json.loads(response.text)
            llm_response_text = json_response['choices'][0]['message']['content']
            self.history.append({
                'role': Role.USER,
                'content': human_message
            })
            self.history.append({
                'role': Role.ASSISTANT,
                'content': llm_response_text
            })
            return response.text
        raise deepseek_exception(response.status_code, response.text)